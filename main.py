# main.py
import os
import yaml
import praw
import logging
import argparse
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RedditRAG:
    """
    Reddit 데이터를 수집, 처리하고 이를 기반으로 질문에 답변하는 RAG 시스템 클래스.
    """
    def __init__(self, config_path="config.yaml"):
        logging.info("RedditRAG 시스템을 초기화합니다...")
        self.config = self._load_config(config_path)
        self._setup_apis()
        
        logging.info(f"임베딩 모델 로딩: {self.config['models']['embedding']}")
        self.embedding_model = SentenceTransformer(self.config['models']['embedding'])
        
        self.generative_model = genai.GenerativeModel(self.config['models']['generative'])
        self.chroma_client = chromadb.PersistentClient(path=self.config['chromadb']['path'])
        
        logging.info("초기화 완료.")

    def _load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_apis(self):
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY가 .env 파일에 없습니다.")
        genai.configure(api_key=google_api_key)

    ### 신규 추가: Gemini 요약 메서드 ###
    def _summarize_post_with_gemini(self, post):
        """Gemini API를 사용하여 게시물 내용을 요약합니다."""
        logging.info(f"게시물 '{post['title'][:30]}...' 요약 중...")
        
        # API 요청 길이를 줄이기 위해 댓글 일부만 사용
        comments_snippet = ' '.join(post['comments'].split()[:300])

        prompt = f"""
        다음 Reddit 게시물의 제목, 본문, 그리고 댓글 요약을 바탕으로 이 게시물의 핵심 주제와 사용자들의 주요 반응을 2~3문장으로 요약해주세요. 모든 답변은 한국어로 작성해주세요.

        - **제목:** {post['title']}
        - **본문:** {post['selftext']}
        - **댓글 일부:** {comments_snippet}
        
        **요약:**
        """
        try:
            response = self.generative_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Gemini 요약 API 호출 중 오류 발생: {e}")
            return "요약 생성에 실패했습니다."

    def _fetch_posts(self, subreddit_name, limit, time_filter):
        logging.info(f"r/{subreddit_name}에서 {time_filter} 기준 상위 {limit}개 게시물을 수집합니다.")
        subreddit = self.reddit.subreddit(subreddit_name)
        top_posts = list(subreddit.top(time_filter=time_filter, limit=limit))
        
        all_posts_data = []
        max_comments = self.config['reddit'].get('max_comments_per_post', None)

        for submission in tqdm(top_posts, desc="게시물 처리 중"):
            comments_text = []
            try:
                submission.comments.replace_more(limit=max_comments)
                comment_count = 0
                for comment in submission.comments.list():
                    if hasattr(comment, 'body'):
                        comments_text.append(comment.body)
                        comment_count += 1
                    if max_comments and comment_count >= max_comments:
                        break

                all_posts_data.append({
                    "id": submission.id,
                    "title": submission.title,
                    "url": submission.url,
                    "selftext": submission.selftext,
                    "comments": ' '.join(comments_text)
                })
            except Exception as e:
                logging.error(f"게시물 {submission.id} 처리 중 오류 발생: {e}")
        return all_posts_data

    ### 수정됨: build_database 메서드에 요약 기능 추가 ###
    def build_database(self, subreddit, limit, time_filter, rebuild=False):
        collection_name = self.config['chromadb']['collection_name']
        
        if rebuild:
            logging.warning(f"기존 컬렉션 '{collection_name}'을 삭제하고 새로 구축합니다.")
            if collection_name in [c.name for c in self.chroma_client.list_collections()]:
                self.chroma_client.delete_collection(name=collection_name)
        
        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        posts_data = self._fetch_posts(subreddit, limit, time_filter)
        
        if not posts_data:
            logging.info("새로운 게시물이 없습니다. 데이터베이스 구축을 종료합니다.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['text_splitter']['chunk_size'],
            chunk_overlap=self.config['text_splitter']['chunk_overlap']
        )
        
        new_chunks_count = 0
        logging.info("="*50)
        for i, post in enumerate(posts_data):
            if not rebuild and collection.get(where={"post_id": post["id"]})['ids']:
                logging.info(f"게시물 ID {post['id']}는 이미 존재하므로 건너뜁니다.")
                continue

            # --- 요약 기능 호출 및 출력 ---
            summary = self._summarize_post_with_gemini(post)
            print(f"\n[게시물 {i+1}/{len(posts_data)} 요약] {post['title']}")
            print(f"URL: {post['url']}")
            print(f"내용 요약: {summary}")
            print("-" * 50)
            # --------------------------------

            document_text = f"Post Title: {post['title']}\n\nURL: {post['url']}\n\nContent: {post['selftext']}\n\n---COMMENTS---\n{post['comments']}"
            chunks = text_splitter.split_text(document_text)
            
            if not chunks:
                continue

            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            metadatas = [{"source_url": post["url"], "title": post["title"], "post_id": post["id"]} for _ in chunks]
            ids = [f"{post['id']}_{i}" for i in range(len(chunks))]
            
            collection.add(embeddings=embeddings.tolist(), documents=chunks, metadatas=metadatas, ids=ids)
            new_chunks_count += len(chunks)

        logging.info(f"데이터베이스 구축 완료! 총 {new_chunks_count}개의 새로운 청크가 저장되었습니다.")
        logging.info(f"현재 컬렉션 '{collection.name}'에는 총 {collection.count()}개의 청크가 있습니다.")


    def chat(self, query):
        collection_name = self.config['chromadb']['collection_name']
        
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except ValueError:
            logging.error(f"DB 컬렉션 '{collection_name}'을 찾을 수 없습니다. 먼저 'build'를 실행하세요.")
            return

        logging.info("관련 문서를 검색합니다...")
        query_embedding = self.embedding_model.encode(query)
        
        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
        context = "\n\n---\n\n".join(results['documents'][0])

        prompt = f"""
        당신은 Reddit 게시물 분석 전문가입니다. 아래 제공된 Reddit 컨텍스트 정보를 바탕으로 사용자의 질문에 대해 친절하고 상세하게 한국어로 답변해주세요.
        컨텍스트에 질문에 대한 정보가 없다면, "제공된 정보만으로는 답변하기 어렵습니다."라고 솔직하게 말해주세요. 절대 정보를 지어내지 마세요.

        --- 컨텍스트 시작 ---
        {context}
        --- 컨텍스트 끝 ---

        사용자 질문: {query}
        """
        
        logging.info("Gemini를 통해 답변을 생성합니다...")
        response = self.generative_model.generate_content(prompt)
        
        print("\n--- 답변 ---")
        print(response.text)
        print("--------------\n")
        
        print("--- 참고 자료 ---")
        sources = {meta['source_url'] for meta in results['metadatas'][0]}
        for url in sources:
            print(f"- {url}")
        print("-----------------")

def main():
    parser = argparse.ArgumentParser(description="Reddit 데이터를 활용한 RAG 챗봇 시스템")
    subparsers = parser.add_subparsers(dest='mode', required=True, help="실행 모드")

    build_parser = subparsers.add_parser('build', help="Reddit 데이터로 벡터 DB를 구축합니다.")
    build_parser.add_argument('--subreddit', type=str, help="데이터를 수집할 서브레딧 이름")
    build_parser.add_argument('--limit', type=int, help="수집할 상위 게시물 수")
    build_parser.add_argument('--rebuild', action='store_true', help="기존 DB를 삭제하고 새로 구축합니다.")
    
    subparsers.add_parser('chat', help="구축된 DB와 대화합니다.")

    args = parser.parse_args()
    
    rag_system = RedditRAG()

    if args.mode == 'build':
        config = rag_system.config['reddit']
        subreddit = args.subreddit if args.subreddit else config['default_subreddit']
        limit = args.limit if args.limit else config['default_limit']
        time_filter = config['time_filter']
        rag_system.build_database(subreddit, limit, time_filter, args.rebuild)

    elif args.mode == 'chat':
        print("\n대화를 시작합니다. 챗봇에게 질문을 입력하세요.")
        print("대화를 종료하려면 '종료' 또는 'exit'를 입력하세요.")
        
        while True:
            try:
                query = input("\n나 > ")
                if query.lower() in ['종료', 'exit']:
                    print("챗봇을 종료합니다. 이용해주셔서 감사합니다.")
                    break
                if not query.strip():
                    continue
                rag_system.chat(query)
            except (KeyboardInterrupt, EOFError):
                print("\n챗봇을 종료합니다.")
                break

if __name__ == "__main__":
    main()