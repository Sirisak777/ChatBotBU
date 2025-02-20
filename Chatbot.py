from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# โหลดข้อมูล
loader = TextLoader("FAQ_BU.txt", encoding="utf-8")  # ใช้ utf-8 
documents = loader.load()

# แยกข้อความออกเป็นส่วนย่อย
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
texts = text_splitter.split_documents(documents)

# สร้าง embeddings และ FAISS index
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = FAISS.from_documents(texts, embeddings)

import google.generativeai as genai

# ตั้งค่า Gemini API Key
GEMINI_API_KEY = "AIzaSyCFTBU-eaJY9PcQVYrMvDJBDKD0eM3mG7s"  #  Gemini API Key
genai.configure(api_key=GEMINI_API_KEY)

# โหลดโมเดล Gemini
gemini_model = genai.GenerativeModel('gemini-pro')

def ask_question(question):
    # ค้นหาข้อมูลที่เกี่ยวข้อง
    relevant_docs = vectorstore.similarity_search(question, k=10)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # ถ้าไม่มีข้อมูลที่เกี่ยวข้อง
    if not context.strip():
        return "ฉันไม่ทราบคำถามของคุณครับ"

    prompt = f"""
    คำถาม: {question}
    ข้อมูลที่เกี่ยวข้อง: {context}

    โปรดวิเคราะห์เจตนาของคำถาม แม้ว่าคำถามจะพิมพ์ไม่ครบประโยคหรือมีคำบางส่วนหายไป หากสามารถตีความได้จากบริบทและข้อมูลที่ให้ กรุณาตอบอย่างกระชับและชัดเจนที่สุด

    - หากคำถามมีคำไม่สมบูรณ์ ให้เดาความหมายตามบริบท
    - เน้นข้อมูลสำคัญและตอบตรงประเด็น
    - ถ้าข้อมูลที่ให้สามารถตอบคำถามได้บางส่วน ให้ตอบเท่าที่ตอบได้
    - ถ้าไม่มีข้อมูลที่เพียงพอ กรุณาตอบว่า "ฉันไม่ทราบคำถามของคุณครับ"

    ตัวอย่าง:
    - คำถาม: ค่าเทอม? → ตอบ: ค่าเทอมเริ่มต้นที่ ... บาท ขึ้นอยู่กับคณะ
    - คำถาม: เดินทาง BU ยังไง → ตอบ: สามารถเดินทางด้วย BTS สถานีหมอชิต, MRT สถานีจตุจักร หรือรถเมล์สาย 39

    โปรดตอบสั้น กระชับ และตรงประเด็น
    """

    # สร้างคำตอบด้วย Gemini
    response = gemini_model.generate_content(prompt)

    # ถ้าคำตอบมีคำว่า "ไม่ทราบ" ให้แสดงแค่ "ไม่ทราบ"
    if "ฉันไม่ทราบคำถามของคุณครับ" in response.text:
        return "ฉันไม่ทราบคำถามของคุณครับ"

    # ให้คำตอบที่แม่นยำและกระชับ
    return response.text.strip()

# รับอินพุตจากผู้ใช้
print("ยินดีต้อนรับสู่ ChatBotBU พิมพ์ 'ขอบคุณ' เพื่อออกจากโปรแกรม")
while True:
    question = input("\nคุณ: ")
    if question.lower() == "ขอบคุณ":
        print("ขอบคุณที่ใช้บริการ ChatBotBU")
        break
    try:
        answer = ask_question(question)
        print(f"ChatBot: {answer}")
    except Exception as e:
        print("ChatBot: ฉันไม่สามารถตอบคำถามนี้ได้ โปรดถามใหม่")