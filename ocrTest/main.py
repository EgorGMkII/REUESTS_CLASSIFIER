import asyncio
import sys
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from openai import AsyncOpenAI
from paddleocr import PaddleOCR
import time
import logging

from document_cleaner import DocumentCleaner
from llm_service import LLMService
from settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
            logging.StreamHandler(sys.stdout)
        ]
)

logger = logging.getLogger("ocr_pipeline")



PDF_DIR = Path("appeals")
OUTPUT_DIR = Path("ocr_results")
OUTPUT_DIR.mkdir(exist_ok=True)


ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="ru",
)



def extract_text_from_page(page, page_num, pdf_name):
    t0 = time.perf_counter()

    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    img_np = np.array(img)

    t1 = time.perf_counter()
    logger.info(f"[{pdf_name}] Page {page_num}: render time = {t1 - t0:.3f}s")

    # OCR
    t2 = time.perf_counter()

    result = ocr.predict(img_np)

    t3 = time.perf_counter()
    logger.info(f"[{pdf_name}] Page {page_num}: OCR time = {t3 - t2:.3f}s")

    texts = []

    for res in result:
        if res.get("rec_texts"):
            texts.extend(res["rec_texts"])

    return "\n".join(texts), (t3 - t0)


async def main():
    global llm_service
    total_start = time.perf_counter()
    cleaner = DocumentCleaner()
    llm_client = AsyncOpenAI(
        api_key=settings.hydra_api_key,
        base_url=settings.hydra_url)
    llm_service = LLMService(llm_client)
    for pdf_file in PDF_DIR.glob("*.pdf"):

        pdf_start = time.perf_counter()
        logger.info(f"START PDF: {pdf_file.name}")

        doc = fitz.open(pdf_file)

        output_text = []
        page_times = []

        for page_num in range(len(doc)):
            logger.info(f"Processing page {page_num + 1}/{len(doc)}")

            page = doc[page_num]

            try:
                page_text, page_time = extract_text_from_page(
                    page,
                    page_num + 1,
                    pdf_file.name
                )

                page_times.append(page_time)

                clean_text, stats = cleaner.clean(page_text)

                output_text.append(
                    f"\n\n{'=' * 50}\n"
                    f"PAGE {page_num + 1}\n"
                    f"{'=' * 50}\n\n"
                    f"{clean_text}"
                )

                logger.info(
                    f"[{pdf_file.name}] Page {page_num + 1} done in {page_time:.3f}s"
                )

            except Exception as e:
                logger.exception(
                    f"[{pdf_file.name}] ERROR on page {page_num + 1}: {e}"
                )

                output_text.append(
                    f"\n\nPAGE {page_num + 1}\nERROR: {e}\n"
                )

        output_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"
        llm_processing_time = time.perf_counter()
        question_type_prediction = await llm_service.choose_question_type(output_text)
        output_text.append('\n\nВид вопроса: '+ str(question_type_prediction))
        logger.info(
            f"LLM processing time for {pdf_file.name}: "
            f"{time.perf_counter() - llm_processing_time:.2f}s"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(output_text))

        pdf_time = time.perf_counter() - pdf_start

        logger.info(
            f"FINISHED PDF: {pdf_file.name} | "
            f"pages={len(doc)} | "
            f"total_time={pdf_time:.2f}s | "
            f"avg_page={sum(page_times) / len(page_times):.2f}s | "
            f"question_type_prediction={question_type_prediction}"
        )

        logger.info(f"Saved: {output_file}")
    total_time = time.perf_counter() - total_start
    logger.info(f"ALL DONE. TOTAL TIME = {total_time:.2f}s")


if __name__ == '__main__':

    asyncio.run(main())