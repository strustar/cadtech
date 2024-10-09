import streamlit as st
import time, os, sys
from cadFcn import extract_images_from_pdf, perform_tesseract_ocr, perform_easyocr, perform_naver_ocr, perform_google_ocr

from langchain_upstage import UpstageLayoutAnalysisLoader
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import nest_asyncio

# Streamlit 페이지 설정
st.set_page_config(page_title="손글씨 추출", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

# 전체 실행 시간 측정 시작
total_start_time = time.time()
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
nest_asyncio.apply()

# 사이드바 설정
with st.sidebar:
    '## :orange[1. PDF 문서 로드]'
    pdf_method = st.selectbox(":green[[PDF 문서 로드 방법 선택]]", ["PyMuPDF", "PDFMiner", "PDFPlumber", "PyPDFium2", "PyPDF2"], index=0)

    "---"
    '## :orange[2. OCR (Multi-Modal) 기반 텍스트 추출]'
    ocr_method = st.radio(":green[[OCR 방법 선택]]", ["Tesseract OCR", "EasyOCR", "Naver Clova OCR", "Google Vision OCR", 'Upstage Layout Analysis Loader', 'Llama Parser', 'All methods'], index=2)

    "---"

st.write("## :green[[PDF 도면 이미지 표시 및 텍스트 추출]]")

uploaded_file = st.sidebar.file_uploader(":blue[다른 PDF 파일을 업로드 할 수 있습니다 (Drag & Drop)]", type=["pdf"])
# 파일이 업로드되지 않았을 때 기본 파일 사용
if uploaded_file is None:
    uploaded_file = "1.pdf"

if isinstance(uploaded_file, str):  # 기본 파일 경로인 경우 (1.pdf)
    file_name = os.path.basename(uploaded_file)
    file_size = os.path.getsize(uploaded_file)
    with open(uploaded_file, "rb") as file:
        images = extract_images_from_pdf(file, pdf_method)
else:  # Streamlit의 UploadedFile 객체인 경우
    file_name = uploaded_file.name
    file_size = uploaded_file.size
    images = extract_images_from_pdf(uploaded_file, pdf_method)
    
st.sidebar.write(f":blue[**PDF 파일 이름**] : :orange[{file_name}]")
st.sidebar.write(f":blue[**파일 크기**] : :orange[{file_size / (1024 * 1024):.2f} MB]")
st.sidebar.write(f":blue[**페이지 수**] : :orange[{len(images)}]")

st.write(f"### :orange[**1. PDF 문서 로더**] : :blue[{pdf_method}]")
st.empty()
if images:
    for i, img in enumerate(images):
        with st.expander(f":blue[{i+1}페이지 도면 이미지] (도면을 보려면 클릭하세요)"):
            st.image(img, use_column_width=False)            
    
        st.write(f"### :orange[**2. OCR (Multi-Modal) 기반 텍스트 추출**] : :blue[{ocr_method}]")
        start_time = time.time()

        if 'Upstage' not in ocr_method and 'Llama' not in ocr_method:            
            if 'Tesseract' in ocr_method:
                text = perform_tesseract_ocr(img)
            if 'EasyOCR' in ocr_method:
                text = perform_easyocr(img)
            if 'Naver Clova OCR' in ocr_method:
                text = perform_naver_ocr(img)
            if 'Google Vision OCR' in ocr_method:
                text = perform_google_ocr(img)
            
            # 줄바꿈을 공백으로 대체하고 여러 공백을 하나로 줄임
            single_line_text = ' '.join(text.split())        
            escaped_text = single_line_text.replace("~", "\~")
            st.write(escaped_text)
        elif 'Upstage' in ocr_method:
            loader = UpstageLayoutAnalysisLoader(
                uploaded_file,
                output_type="text",  # output_type: 출력 형식 [(기본값)'html', 'text']
                split="element",    # 문서 분할 방식 ['none', 'element', 'page']
                use_ocr=True,
                # exclude=["header", "footer"],
            )
            docs = loader.load()
            for doc in docs:
                st.write(doc.page_content)
        elif 'Llama' in ocr_method:
            parsing_instruction = (
                "You are parsing a brief of AI Report. Please extract tables in markdown format."
            )
            documents = LlamaParse(
                use_vendor_multimodal_model=True,
                vendor_multimodal_model_name="openai-gpt4o",
                vendor_multimodal_api_key=os.environ["OPENAI_API_KEY"],
                result_type="markdown",
                language="ko",
                parsing_instruction=parsing_instruction,
                # skip_diagonal_text=True,
                # page_separator="\n=================\n"
            )
            # langchain 도큐먼트로 변환
            # parsing 된 결과
            parsed_docs = documents.load_data(file_path=uploaded_file)
            docs = [doc.to_langchain_format() for doc in parsed_docs]
            # markdown 형식으로 추출된 테이블 확인
            st.write(docs[0].page_content)




            # 파서 설정
            parser = LlamaParse(
                result_type="markdown",  # "markdown"과 "text" 사용 가능
                num_workers=8,  # worker 수 (기본값: 4)
                verbose=True,
                language="ko",
            )

            # SimpleDirectoryReader를 사용하여 파일 파싱
            file_extractor = {".pdf": parser}

            # LlamaParse로 파일 파싱
            documents = SimpleDirectoryReader(
                input_files=[uploaded_file],
                file_extractor=file_extractor,
            ).load_data()
            # 랭체인 도큐먼트로 변환
            docs = [doc.to_langchain_format() for doc in documents]
            for doc in docs:
                st.write(doc.page_content)

        end_time = time.time()
        st.write(f"{ocr_method} 처리 시간 : {end_time - start_time:.2f} 초")   # st.text_area(f"Tesseract OCR 결과 (이미지 {i+1}, {pdf_method}):", text, height=250)
    
else:
    st.warning(f"{pdf_method} 방법으로 이미지를 추출할 수 없습니다.")


# 전체 실행 시간 계산 및 표시
total_end_time = time.time()
st.write(f"전체 실행 시간: {total_end_time - total_start_time:.2f} 초")


