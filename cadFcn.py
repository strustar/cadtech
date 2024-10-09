import streamlit as st
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTImage, LTFigure
import pypdfium2 as pdfium
import PyPDF2, pdfplumber, fitz  # PyMuPDF
import io, time, uuid, json, requests, os
import pytesseract
import easyocr
import numpy as np
from PIL import Image as PILImage
from google.cloud.vision_v1.types import Image as VisionImage
from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageRequest, ImageContext

def perform_google_ocr(image):
    # 환경 변수가 올바르게 설정되었는지 확인
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_path:
        (f"Credentials file path: {credentials_path}")
    else:
        ("GOOGLE_APPLICATION_CREDENTIALS is not set.")

    # 환경 변수 설정
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vision-api-key.json"

    # Vision API 클라이언트 초기화
    client = vision.ImageAnnotatorClient()

    # PIL 이미지를 바이트로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Google Vision Image 객체 생성
    image = VisionImage(content=img_byte_arr)

    # 이미지 컨텍스트 설정
    image_context = ImageContext(
        language_hints=['ko', 'en'],  # 한국어 힌트 추가
    )

    # 텍스트 추출 요청 설정
    request = AnnotateImageRequest(
        image=image,
        image_context=image_context,
        features=[{'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]  # DOCUMENT_TEXT_DETECTION 사용
    )

    try:
        # 텍스트 추출 요청
        response = client.annotate_image(request)

        # 추출된 텍스트 처리
        if response.full_text_annotation:
            full_text = response.full_text_annotation.text
            
            # 페이지 및 블록 정보 추출 (선택사항)
            pages = []
            for page in response.full_text_annotation.pages:
                blocks = []
                for block in page.blocks:
                    block_text = ' '.join([
                        ' '.join([word.symbols[0].text + ''.join([symbol.text for symbol in word.symbols[1:]])
                                for word in paragraph.words])
                        for paragraph in block.paragraphs
                    ])
                    blocks.append({
                        'text': block_text,
                        'confidence': block.confidence,
                        'bounding_box': [(vertex.x, vertex.y) for vertex in block.bounding_box.vertices]
                    })
                pages.append(blocks)
        else:
            full_text = "텍스트를 찾을 수 없습니다."
            pages = []

    except Exception as e:
        full_text = f"OCR 처리 중 오류 발생: {str(e)}"
        pages = []

    return full_text #, pages


def perform_tesseract_ocr(image):
    """Tesseract OCR을 사용하여 텍스트를 추출합니다."""
    # Tesseract 실행 파일 경로 설정
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'    
    text = pytesseract.image_to_string(image, lang='kor+eng', config='--oem 3 --psm 11 --dpi 300')    
    return text.strip()

def perform_easyocr(image, gpu=True, batch_size=4, paragraph=False, min_size=10, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003):
    """EasyOCR을 사용하여 텍스트를 추출하고 성능을 개선합니다."""
    # EasyOCR 리더 초기화
    reader = easyocr.Reader(['ko', 'en'], gpu=True, quantize=True)
    
    result = reader.readtext(
        np.array(image),
        batch_size=batch_size,
        paragraph=paragraph,
        min_size=min_size,
        contrast_ths=contrast_ths,
        adjust_contrast=adjust_contrast,
        filter_ths=filter_ths
    )    
    
    text = ' '.join([text for _, text, _ in result])    
    return text

def perform_naver_ocr(image):
    """Naver Clova OCR API를 사용하여 텍스트를 추출합니다."""
    # Naver Clova OCR API 설정
    api_url = 'https://vqxu0j6v9h.apigw.ntruss.com/custom/v1/34752/25fcd529b5366044e5b35700d8f5ceeff282092b3bd3308ddb0faf41a801bdd2/general'
    secret_key = 'TWxuTG5kSklPTFVGeWtJRkxVeVVvZWFwZmhqSFNOTGQ='
    
    # 이미지를 임시 파일로 저장
    temp_image_path = f"temp_image_{uuid.uuid4()}.png"
    image.save(temp_image_path)
    
    request_json = {
        'images': [{'format': 'png', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    
    extracted_text = []
    try:
        with open(temp_image_path, 'rb') as f:
            files = [('file', f)]
            headers = {'X-OCR-SECRET': secret_key}
            response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
        
        result = response.json()
        # st.write(f"API 응답: {result}")  # 전체 응답 출력
        # # 원하는 정보만 출력
        # for field in result['images'][0]['fields']:
        #     (f"텍스트: {field['inferText']}")
        #     (f"경계상자: {field['boundingPoly']}")
        #     (f"줄바꿈: {field['lineBreak']}")
        #     ("---")

        # API 응답 확인 및 에러 처리
        if response.status_code != 200:
            st.error(f"API 호출 실패: 상태 코드 {response.status_code}")
            return "API 호출 실패"

        if 'images' not in result:
            st.error("API 응답에 'images' 키가 없습니다.")
            return "API 응답 형식 오류"

        if len(result['images']) == 0:
            st.warning("OCR 결과가 없습니다.")
            return "OCR 결과 없음"

        if 'fields' not in result['images'][0]:
            st.warning("OCR 결과에 텍스트 필드가 없습니다.")
            return "텍스트 필드 없음"

        for field in result['images'][0]['fields']:
            extracted_text.append(field['inferText'])

    except Exception as e:
        st.error(f"OCR 처리 중 오류 발생: {str(e)}")
        print(f"예외 발생: {str(e)}")  # 예외 정보 출력
        return f"오류: {str(e)}"
    
    finally:
        # 임시 파일 삭제 시도
        try:
            os.remove(temp_image_path)
        except PermissionError:
            print(f"Warning: Unable to delete temporary file {temp_image_path}")    
    
    return ' '.join(extracted_text)



def extract_images_from_pdf(file_path, pdf_method):
    """PDF 파일에서 이미지를 추출합니다. 선택된 방법에 따라 추출합니다."""
    images = []
    if pdf_method == "PyMuPDF":
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = PILImage.open(io.BytesIO(image_bytes)).convert('RGB')
                images.append(image)
        pdf_document.close()
    elif pdf_method == "PyPDF2":
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                            data = xObject[obj].get_data()
                            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                                mode = "RGB"
                            else:
                                mode = "P"
                            
                            if '/Filter' in xObject[obj]:
                                if xObject[obj]['/Filter'] == '/FlateDecode':
                                    img = PILImage.frombytes(mode, size, data)
                                elif xObject[obj]['/Filter'] == '/DCTDecode':
                                    img = PILImage.open(io.BytesIO(data))
                                elif xObject[obj]['/Filter'] == '/JPXDecode':
                                    img = PILImage.open(io.BytesIO(data))
                                else:
                                    continue
                            else:
                                img = PILImage.frombytes(mode, size, data)
                            images.append(img)
    elif pdf_method == "PDFMiner":
        for page_layout in extract_pages(file_path):
            for element in page_layout:
                if isinstance(element, LTImage):
                    try:
                        image_data = element.stream.get_data()
                        image = PILImage.open(io.BytesIO(image_data))
                        images.append(image)
                    except Exception as e:
                        st.warning(f"PDFMiner: LTImage 처리 중 오류 발생 - {str(e)}")
                elif isinstance(element, LTFigure):
                    try:
                        for fig_element in element:
                            if isinstance(fig_element, LTImage):
                                try:
                                    image_data = fig_element.stream.get_data()
                                    image = PILImage.open(io.BytesIO(image_data))
                                    images.append(image)
                                except Exception as e:
                                    st.warning(f"PDFMiner: LTFigure 내 LTImage 처리 중 오류 발생 - {str(e)}")
                    except Exception as e:
                        st.warning(f"PDFMiner: LTFigure 처리 중 오류 발생 - {str(e)}")
    elif pdf_method == "PDFPlumber":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                for img in page.images:
                    image = PILImage.open(io.BytesIO(img['stream'].get_data()))
                    images.append(image)
    elif pdf_method == "PyPDFium2":
        pdf = pdfium.PdfDocument(file_path)
        n_pages = len(pdf)
        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            pil_image = page.render().to_pil()
            images.append(pil_image)    
    return images