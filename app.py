# ==============================================================================
# GEREKLİ KÜTÜPHANELER (app.py)
# Amaç: Streamlit web arayüzü ve RAG zinciri için gerekli tüm
#       Python kütüphanelerini içe aktarmak.
# ==============================================================================
import streamlit as st  # Web arayüzünü (butonlar, metin kutuları vb.) oluşturmak için
import os               # İşletim sistemiyle etkileşim için (API anahtarını ayarlamak, dosya yolunu kontrol etmek)
import pandas as pd       # (Bu kodda doğrudan kullanılmıyor ama genellikle veri göstermek için eklenir)
from tqdm import tqdm     # (Bu kodda doğrudan kullanılmıyor, Colab'de kullanılmıştı)

# --- LangChain Kütüphaneleri ---
from langchain_text_splitters import RecursiveCharacterTextSplitter # (Bu dosyada gerek yok ama zararı da yok)
from langchain_core.documents import Document                     # (Bu dosyada gerek yok ama zararı da yok)
from langchain_chroma import Chroma                     # Diskten 'chroma_db' veritabanını yüklemek için
from langchain_huggingface import HuggingFaceEmbeddings # Veritabanını yüklerken hangi embedding modelini kullandığımızı belirtmek için
from langchain_google_genai import ChatGoogleGenerativeAI # RAG zincirinin beyni olan LLM'i (Gemini) yüklemek için
from langchain_core.prompts import PromptTemplate       # LLM'e vereceğimiz talimat şablonu (Prompt) için
from langchain_core.runnables import RunnablePassthrough # Zincir içinde kullanıcı sorusunu değiştirmeden aktarmak için
from langchain_core.output_parsers import StrOutputParser # LLM'in cevabını temiz bir metne (string) dönüştürmek için

# ==============================================================================
# YENİ EKLENDİ: KENAR ÇUBUĞU (SIDEBAR) HAKKINDA KUTUSU
# ==============================================================================
st.sidebar.title("Proje Hakkında ℹ️")
st.sidebar.info(
    "Bu chatbot, Akbank GenAI Bootcamp projesi için geliştirilmiştir.\n\n"
    "RAG (Retrieval-Augmented Generation) mimarisi kullanılarak, "
    "şarkı sözlerinin bir kısmını yazdığınızda o şarkının adını, "
    "sanatçısını ve türünü bulur."
)
st.sidebar.markdown("---") # Ayırıcı çizgi

# ==============================================================================
# GOOGLE API ANAHTARI AYARI (DEPLOY İÇİN SON VERSİYON: TRY-EXCEPT)
# ==============================================================================

# Bu fonksiyon, en basit ve en sağlam yöntemi kullanır:
# 1. 'st.secrets'tan anahtarı okumayı Dener (try).
# 2. Başarılı olursa (Cloud'dayız demektir), anahtarı döndürür.
# 3. Hata alırsa (Lokal'deyiz demektir),
#    kenar çubuğundaki metin kutusunu gösterir.

def get_google_api_key():
    try:
        # Önce Streamlit Cloud Secrets'ı okumayı DENE
        return st.secrets["GOOGLE_API_KEY"]
    except:
        # Eğer 'secrets' yoksa (Lokal) veya anahtar tanımlı değilse:
        # Kenar çubuğundan (Lokal) anahtarı iste
        return st.sidebar.text_input(
            "Google API Anahtarınızı Buraya Yapıştırın:",
            type="password",
            help="API anahtarınızı Google AI Studio'dan alabilirsiniz."
        )

# Fonksiyonu çağır ve anahtarı al
api_key = get_google_api_key()

# Anahtarın alınıp alınmadığını kontrol et
if api_key:
    # Anahtarı, LangChain kütüphanelerinin otomatik olarak okuyacağı
    # 'os.environ' (ortam değişkeni) içine ata.
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    # Eğer kullanıcı henüz bir anahtar girmemişse (veya Secrets'ta da yoksa):
    # Sol kenar çubuğuna bir uyarı koy
    st.sidebar.warning("Lütfen sol kenar çubuğundan Google API anahtarlarınızı girin.")
    # Anahtarı girmeden uygulamanın geri kalanının çalışmasını durdur
    st.stop()
# ==============================================================================
# RAG ZİNCİRİNİ YÜKLEME FONKSİYONU (Hücre 4'ün Gelişmiş Hali)
# Amaç: RAG zincirini (Retriever, Prompt, LLM) kuran ana fonksiyon.
# ==============================================================================

# @st.cache_resource: Bu, Streamlit için HAYATİ bir optimizasyon komutudur.
@st.cache_resource
def load_rag_chain():
    """
    (Fonksiyon açıklaması - Docstring)
    Colab'de oluşturulan 'chroma_db' veritabanını diskten yükler
    ve RAG zincirini kurar.
    """
    # Kullanıcıya modelin yüklendiğini bildiren bir durum mesajı
    st.sidebar.info("RAG Zinciri ve Model Yükleniyor...")
    
    # --- 1. Retriever'ı Kur (Veritabanı Yükleme) ---
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # 'app.py'nin çalıştığı dizinde 'chroma_db' adında bir klasör ara.
    if not os.path.exists("chroma_db"):
        # Eğer klasör bulunamazsa:
        # Ana ekrana büyük bir HATA mesajı bas.
        st.error(
            "HATA: 'chroma_db' klasörü bulunamadı! "
            "Lütfen Colab'den indirdiğiniz 'chroma_db' klasörünü 'app.py' dosyasının yanına kopyalayın."
        )
        st.stop() # Uygulamayı durdur
        
    # 'chroma_db' klasörünü diskten yükle ve 'vectorstore' nesnesini oluştur.
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    # Optimize ettiğimiz en iyi ayar (Colab Hücre 4'teki gibi):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) 
    
    # --- 2. Prompt'u Kur (Colab Hücre 4'teki gibi) ---
    template = """
    Sana verilen Bağlam'ı kullanarak kullanıcının Soru'sunu cevapla. Cevabını SADECE bu bağlamdaki bilgilere dayanarak ver. Eğer bağlamda cevap yoksa, 'Bu konuda bilgim yok.' de.
    Bağlam:
    {context}
    Soru: {question}
    Cevap:
    """
    prompt = PromptTemplate.from_template(template)

    # --- 3. LLM'i Kur (Colab Hücre 4'teki gibi) ---
    llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)

    # --- 4. Zinciri (Chain) Oluştur (Colab Hücre 4'teki gibi) ---
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # Adım 1: Context'i bul, soruyu aktar
        | prompt                                                   # Adım 2: Prompt'u doldur
        | llm                                                      # Adım 3: LLM'e sor
        | StrOutputParser()                                        # Adım 4: Cevabı metne çevir
    )
    
    # Model yüklemesi bittiğinde kenar çubuğundaki 'info' mesajını
    # 'success' (başarı) mesajı ile değiştir.
    st.sidebar.success("RAG Zinciri ve Model Başarıyla Yüklendi!")
    return rag_chain # Hafızaya (cache) alınacak olan RAG zincirini döndür.

# ==============================================================================
# WEB ARAYÜZÜ (STREAMLIT) - TASARIM GÜNCELLENDİ
# ==============================================================================

# YENİ EKLENDİ: Başlığı ortalamak ve stilleri iyileştirmek için HTML/CSS
st.markdown("""
<style>
h1 {
    font-size: 2.0rem   !important; /* Başlığı biraz büyüttüm */
    text-align: center;          /* BAŞLIĞI ORTALA */
    padding-bottom:20px;         /* Altına biraz boşluk */
    margin-bottom:80px;         /* Çizginin altına boşluk */
}

/* YENİ EKLENDİ: Merhaba yazısını aşağı itmek için */
.welcome-text {
    margin-top: 30px; /* Başlıktaki 80px boşluğa EK olarak 30px daha boşluk ekler */
    font-size: 1.1rem; /* Yazıyı biraz büyütelim */
}

/* Metin giriş kutusunun görünümünü iyileştir */
.stTextInput > div > div > input {
    font-size: 1.1rem;
    padding-top: 10px;
    padding-bottom: 10px;
}
</style>

<h1>🎵 Şarkı Sözlerinden Şarkı Bulma Chatbot'u 🎵</h1>
""", unsafe_allow_html=True)

# DEĞİŞTİ: 'Merhaba...' satırını da st.markdown içine alıp sınıfı uyguluyoruz
st.markdown('<p class="welcome-text">Merhaba, nasıl yardımcı olabilirim?</p>', unsafe_allow_html=True)

# --- Ana Uygulama Mantığı ---
try:
    # ADIM 1: RAG zincirini yükle.
    chain = load_rag_chain()
    
    # ADIM 2: Kullanıcıdan soru almak için bir metin giriş kutusu oluştur.
    # YENİ EKLENDİ: 'placeholder' (silik örnek metin) eklendi
    user_question = st.text_input(
        "Hadi bana bulmak istediğin şarkının sözlerini söyle...",
        placeholder=""
    )
    
    # YENİ EKLENDİ: Örnek soruları gösteren bir alt başlık
    st.caption("Örnek kullanım: 'Aşkın kazanması için ayrı gitme' sözü geçen şarkı hangisi, kim söylüyor ve türü nedir?")

    # ADIM 3: Eğer kullanıcı bir soru yazıp 'Enter'a bastıysa...
    if user_question:
        # 'st.spinner', kullanıcıya "bekle" mesajı gösteren
        # dönen bir animasyon başlatır.
        with st.spinner("Şarkı aranıyor... 🎶"):
            # RAG zincirini kullanıcının sorusuyla çalıştır (invoke).
            cevap = chain.invoke(user_question) 
        
        # 'with st.spinner' bloğu bittiğinde (cevap alındığında)
        # animasyon otomatik olarak kaybolur.
        
        # ADIM 4: Gelen cevabı kontrol et.
        if "Bu konuda bilgim yok" in cevap:
            # Başarısızlık durumu: LLM bir şey bulamadı.
            st.error("Üzgünüm bulamadım. Farklı bir söz dizisi deneyebilirsin.")
        else:
            # Başarı durumu: LLM bir cevap buldu.
            st.success("İşte buldum!") 
            st.markdown(f"**Cevap:** {cevap}")

except Exception as e:
    # 'load_rag_chain' veya 'chain.invoke' sırasında bir hata oluşursa
    st.error(f"Bir hata oluştu: {e}")
    st.error("Lütfen 'chroma_db' klasörünün doğru yerde olduğundan veya API anahtarınızın doğru olduğundan emin olun.")