# Smart Pricing & Profit AI Assistant
### دنيا الأنظمة لتقنية المعلومات

## نظرة عامة
تطبيق ذكي يساعد الشركات على:
- حساب الربح الحقيقي لكل منتج
- مراقبة تغيرات أسعار الموردين
- اقتراح سعر البيع المناسب
- تحليل ربح كل عميل
- الإجابة على الأسئلة التجارية عبر شات بوت مدعوم بـ NLP

## المسار الأكاديمي
**Track 1: Natural Language Processing (NLP)**
- TF-IDF vectorizer (character n-grams)
- Logistic Regression classifier
- Arabic text normalization

## تشغيل المشروع

```bash
# 1. إنشاء بيئة افتراضية
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. تثبيت المكتبات
pip install -r requirements.txt

# 3. تشغيل التطبيق
streamlit run app.py
```

## هيكل المشروع
```
smart_pricing_ai/
├── app.py                  # نقطة الدخول الرئيسية
├── requirements.txt
├── data/                   # ملفات CSV
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── calculations.py
│   ├── pricing.py
│   ├── supplier_monitor.py
│   ├── client_analysis.py
│   └── chatbot.py          # نموذج TF-IDF + Logistic Regression
└── models/
    └── intent_examples.json
```

## الصفحات
1. **Dashboard** — لوحة تحكم رئيسية بالأرباح والإيرادات
2. **Profit Analysis** — تحليل تفصيلي للعملاء والمنتجات
3. **Supplier Monitor** — تنبيهات تغيرات أسعار الموردين
4. **Pricing Calculator** — حاسبة سعر البيع بهامش مستهدف
5. **Chat Assistant** — شات بوت NLP للأسئلة التجارية
6. **Model Evaluation** — تقرير أداء النموذج (accuracy, F1)
