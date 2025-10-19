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
# Colab notebook'unda RAG zincirini kurarken kullandığımız
# tüm temel bileşenleri buraya da import ediyoruz.
from langchain_text_splitters import RecursiveCharacterTextSplitter # (Bu dosyada gerek yok ama zararı da yok)
from langchain_core.documents import Document                     # (Bu dosyada gerek yok ama zararı da yok)
from langchain_chroma import Chroma                     # Diskten 'chroma_db' veritabanını yüklemek için
from langchain_huggingface import HuggingFaceEmbeddings # Veritabanını yüklerken hangi embedding modelini kullandığımızı belirtmek için
from langchain_google_genai import ChatGoogleGenerativeAI # RAG zincirinin beyni olan LLM'i (Gemini) yüklemek için
from langchain_core.prompts import PromptTemplate       # LLM'e vereceğimiz talimat şablonu (Prompt) için
from langchain_core.runnables import RunnablePassthrough # Zincir içinde kullanıcı sorusunu değiştirmeden aktarmak için
from langchain_core.output_parsers import StrOutputParser # LLM'in cevabını temiz bir metne (string) dönüştürmek için
# ==============================================================================
# GOOGLE API ANAHTARI AYARI (LOKAL/DEPLOYMENT İÇİN DÜZELTİLDİ)
# ==============================================================================

# Bu fonksiyon, UYGULAMANIN NEREDE ÇALIŞTIĞINI KONTROL EDER.
# Streamlit Cloud'daysa, 'st.secrets'tan anahtarı okur.
# Lokal (senin bilgisayarın) ise, kenar çubuğunda metin kutusunu gösterir.

def get_google_api_key():
    # 'os.environ.get' ile ortam değişkenlerini kontrol ediyoruz.
    # "STREAMLIT_SHARING_MODE" değişkeni SADECE Streamlit Cloud'da "True" olur.
    # Bu kontrol, st.secrets'ı lokalde aramayı engeller ve çökmeyi önler.
    if os.environ.get('STREAMLIT_SHARING_MODE') == 'True':
        # Streamlit Cloud'da çalışıyoruz, Secrets'tan anahtarı al
        try:
            return st.secrets["GOOGLE_API_KEY"]
        except KeyError:
            # Secrets'ta anahtarın unutulması durumuna karşı bir hata
            st.error("HATA: Streamlit Cloud Secrets'ta 'GOOGLE_API_KEY' bulunamadı!")
            st.stop()
    else:
        # Lokal'de çalışıyoruz, kenar çubuğundan anahtarı iste
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
    # Eğer kullanıcı henüz bir anahtar girmemişse (ve Secrets'ta da yoksa):
    # Sol kenar çubuğuna bir uyarı koy
    st.sidebar.warning("Lütfen sol kenar çubuğundan Google API anahtarlarınızı girin.")
    # Anahtarı girmeden uygulamanın geri kalanının çalışmasını durdur
    st.stop()
# ==============================================================================
# RAG ZİNCİRİNİ YÜKLEME FONKSİYONU (Hücre 4'ün Gelişmiş Hali)
# Amaç: RAG zincirini (Retriever, Prompt, LLM) kuran ana fonksiyon.
# ==============================================================================

# @st.cache_resource: Bu, Streamlit için HAYATİ bir optimizasyon komutudur.
# Bu 'decorator', altındaki 'load_rag_chain' fonksiyonunun SADECE BİR KEZ
# çalışmasını ve sonucunu (kurulan 'rag_chain') hafızada (cache) tutmasını sağlar.
#
# EĞER BU OLMASAYDI: Kullanıcı her soru sorduğunda, Streamlit tüm 'app.py'
# dosyasını baştan çalıştırır ve her seferinde modeli + veritabanını
# yeniden yüklerdi. Bu da saniyeler süren yavaş bir uygulama demekti.
# @st.cache_resource sayesinde model ve DB bir kez yüklenir, anında cevap alınır.
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
    # Colab'de 'chroma_db'yi oluştururken kullandığımız modelin aynısı olmalı.
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
    # Kullanıcının sorusuna en çok benzeyen 'k=1' (en iyi 1) sonucu getir.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) 
    
    # --- 2. Prompt'u Kur (Colab Hücre 4'teki gibi) ---
    # LLM'e (Gemini) vereceğimiz talimat şablonu.
    template = """
    Sana verilen Bağlam'ı kullanarak kullanıcının Soru'sunu cevapla. Cevabını SADECE bu bağlamdaki bilgilere dayanarak ver. Eğer bağlamda cevap yoksa, 'Bu konuda bilgim yok.' de.
    Bağlam:
    {context}
    Soru: {question}
    Cevap:
    """
    prompt = PromptTemplate.from_template(template)

    # --- 3. LLM'i Kur (Colab Hücre 4'teki gibi) ---
    # 'gemini-pro-latest' günlük 50 limitine takılabilir.
    llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)

    # --- 4. Zinciri (Chain) Oluştur (Colab Hücre 4'teki gibi) ---
    # LangChain Expression Language (LCEL) kullanarak boru hattını tanımla.
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
# WEB ARAYÜZÜ (STREAMLIT)
# Amaç: Kullanıcının göreceği ve etkileşime gireceği arayüzü çizdirmek.
# ==============================================================================

# st.markdown: Ekrana metin (veya HTML) basar.
# 'unsafe_allow_html=True', <style> ve <h1> gibi HTML etiketlerini
# kullanmamıza izin verir (Başlığı küçültmek için).
st.markdown("""
<style>
h1 {
    font-size: 2.0rem !important;
}
</style>
<h1>🎵 Şarkı Sözlerinden Şarkı Bulma Chatbot'u 🎵</h1>
""", unsafe_allow_html=True)

st.markdown("Merhaba, nasıl yardımcı olabilirim?")

# --- Ana Uygulama Mantığı ---
# Bu 'try-except' bloğu, RAG zinciri yüklenirken veya çalışırken
# (örn: bozuk veritabanı, yanlış API anahtarı) oluşabilecek
# herhangi bir hatayı yakalar ve uygulamanın çökmesini engeller.
try:
    # ADIM 1: RAG zincirini yükle.
    # (Bu fonksiyon @st.cache_resource sayesinde sadece ilk seferde
    # gerçekten çalışır, sonraki seferlerde hafızadan (cache) çağrılır.)
    chain = load_rag_chain()
    
    # ADIM 2: Kullanıcıdan soru almak için bir metin giriş kutusu oluştur.
    user_question = st.text_input("Bulmak istediğiniz şarkı sözünü yazın:")

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
        # Bu, RAG prompt'umuzdaki ('Bu konuda bilgim yok.') kuralı
        # yakalamak için çok önemli bir kullanıcı deneyimi (UX) adımıdır.
        if "Bu konuda bilgim yok" in cevap:
            # Başarısızlık durumu: LLM bir şey bulamadı.
            # st.error: Kırmızı bir hata kutusu gösterir.
            st.error("Üzgünüm bulamadım. Farklı bir söz dizisi deneyebilirsin.")
        else:
            # Başarı durumu: LLM bir cevap buldu.
            # st.success: Yeşil bir başarı kutusu gösterir.
            st.success("İşte buldum!") 
            # st.markdown: Cevabı kalın (**) olarak ekrana basar.
            st.markdown(f"**Cevap:** {cevap}")

except Exception as e:
    # 'load_rag_chain' veya 'chain.invoke' sırasında bir hata oluşursa
    # (örn: veritabanı bozuksa, API anahtarı geçersizse),
    # bu blok çalışır ve kullanıcıya net bir hata mesajı gösterir.
    st.error(f"Bir hata oluştu: {e}")
    st.error("Lütfen 'chroma_db' klasörünün doğru yerde olduğundan veya API anahtarınızın doğru olduğundan emin olun.")