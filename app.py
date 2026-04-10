import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.data_loader    import load_data
from src.preprocessing  import clean_dataframes
from src.calculations   import build_merged_df
from src.client_analysis import client_profit_report, best_client
from src.supplier_monitor import all_supplier_changes
from src.chatbot        import detect_intent, extract_name_after_keyword, train_intent_model, get_confidence
from src.pricing        import suggest_selling_price, price_sensitivity
from src.utils          import fmt_currency, fmt_percent
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Pricing & Profit AI",
    page_icon="💹",
    layout="wide",
)

# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def get_data():
    products, purchases, sales, clients, suppliers = load_data()
    products, purchases, sales, clients, suppliers = clean_dataframes(
        products, purchases, sales, clients, suppliers
    )
    merged = build_merged_df(products, purchases, sales)
    return products, purchases, sales, clients, suppliers, merged

products, purchases, sales, clients, suppliers, merged = get_data()

@st.cache_resource
def get_model_metrics():
    return train_intent_model()

model_metrics = get_model_metrics()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💹 Smart Pricing AI")
    st.caption("World Systems for IT — دنيا الأنظمة")
    st.divider()
    page = st.radio(
        "الصفحات",
        ["🏠 Dashboard", "📊 Profit Analysis", "🚨 Supplier Monitor",
         "🧮 Pricing Calculator", "🤖 Chat Assistant", "🔬 Model Evaluation"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"نموذج NLP — دقة: **{model_metrics['accuracy']}%**")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🏠 لوحة التحكم الرئيسية")

    total_profit  = merged["profit"].sum()
    total_revenue = (merged["selling_price"] * merged["quantity"]).sum()
    avg_margin    = merged["margin"].mean()
    low_margin_n  = (merged["margin"] < 10).sum()

    best = best_client(merged)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 إجمالي الأرباح",   fmt_currency(total_profit))
    c2.metric("📦 إجمالي الإيرادات", fmt_currency(total_revenue))
    c3.metric("📈 متوسط الهامش",     fmt_percent(avg_margin))
    c4.metric("⚠️ منتجات هامش ضعيف", f"{low_margin_n} منتج")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("أرباح العملاء")
        client_report = client_profit_report(merged)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(client_report["client_name"], client_report["total_profit"],
                color=["#1D9E75", "#378ADD", "#D85A30", "#BA7517"])
        ax.set_xlabel("الربح (SAR)")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("توزيع الهوامش")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.hist(merged["margin"], bins=10, color="#378ADD", edgecolor="white")
        ax2.set_xlabel("هامش الربح %")
        ax2.set_ylabel("عدد المبيعات")
        plt.tight_layout()
        st.pyplot(fig2)

    st.divider()
    st.subheader("📋 جدول المبيعات الكامل")
    st.dataframe(
        merged[["product_name", "client_name", "selling_price",
                "total_cost", "profit", "margin", "quantity", "sale_date"]]
        .rename(columns={
            "product_name":  "المنتج",
            "client_name":   "العميل",
            "selling_price": "سعر البيع",
            "total_cost":    "التكلفة الكاملة",
            "profit":        "الربح",
            "margin":        "الهامش %",
            "quantity":      "الكمية",
            "sale_date":     "تاريخ البيع",
        }),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PROFIT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Profit Analysis":
    st.title("📊 تحليل الأرباح")

    tab1, tab2 = st.tabs(["تحليل العملاء", "تحليل المنتجات"])

    with tab1:
        report = client_profit_report(merged)
        st.dataframe(
            report.rename(columns={
                "client_name":       "العميل",
                "total_profit":      "إجمالي الربح",
                "total_revenue":     "إجمالي الإيرادات",
                "num_transactions":  "عدد المعاملات",
                "avg_margin":        "متوسط الهامش %",
            }),
            use_container_width=True,
        )

    with tab2:
        prod_profit = (
            merged.groupby("product_name")["profit"]
            .sum().reset_index()
            .sort_values("profit", ascending=False)
        )
        st.dataframe(
            prod_profit.rename(columns={"product_name": "المنتج", "profit": "إجمالي الربح"}),
            use_container_width=True,
        )
        low = merged[merged["margin"] < 15]
        if not low.empty:
            st.warning(f"⚠️ {len(low)} مبيعات بهامش أقل من 15%")
            st.dataframe(low[["product_name", "client_name", "margin", "profit"]])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SUPPLIER MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Supplier Monitor":
    st.title("🚨 مراقبة أسعار الموردين")

    changes = all_supplier_changes(purchases)
    if changes.empty:
        st.info("لا تتوفر بيانات كافية للمقارنة.")
    else:
        changes = changes.merge(products[["product_id", "product_name"]], on="product_id", how="left")
        for _, row in changes.iterrows():
            direction = row["direction"]
            icon      = "🔴" if direction == "up" else "🟢" if direction == "down" else "🟡"
            pct       = row["change_percent"]
            st.metric(
                label=f"{icon} {row.get('product_name', row['product_id'])}",
                value=fmt_currency(row["new_price"]),
                delta=f"{pct:+.1f}% من {fmt_currency(row['old_price'])}",
            )
        st.divider()
        st.dataframe(changes.rename(columns={
            "product_name":   "المنتج",
            "old_price":      "السعر القديم",
            "new_price":      "السعر الجديد",
            "change_percent": "نسبة التغير %",
            "direction":      "الاتجاه",
        }), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PRICING CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Pricing Calculator":
    st.title("🧮 حاسبة سعر البيع")

    col1, col2 = st.columns(2)
    with col1:
        purchase  = st.number_input("سعر الشراء (SAR)", min_value=0.0, value=1000.0, step=10.0)
        tax       = st.number_input("الضريبة (SAR)",   min_value=0.0, value=150.0,  step=5.0)
        shipping  = st.number_input("الشحن (SAR)",     min_value=0.0, value=50.0,   step=5.0)
        expenses  = st.number_input("مصاريف أخرى",    min_value=0.0, value=20.0,   step=5.0)
    with col2:
        target_margin = st.slider("هامش الربح المستهدف %", 5, 60, 20)
        total_cost    = purchase + tax + shipping + expenses
        suggested     = suggest_selling_price(total_cost, target_margin)
        st.metric("التكلفة الكاملة",    fmt_currency(total_cost))
        st.metric("سعر البيع المقترح", fmt_currency(suggested))
        st.metric("الربح المتوقع",      fmt_currency(suggested - total_cost))

    st.divider()
    st.subheader("جدول الحساسية — أسعار عند هوامش مختلفة")
    sensitivity = price_sensitivity(total_cost)
    df_sens = pd.DataFrame([
        {"الهامش %": f"{m}%", "سعر البيع": fmt_currency(p), "الربح": fmt_currency(p - total_cost)}
        for m, p in sensitivity.items()
    ])
    st.dataframe(df_sens, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: CHAT ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Chat Assistant":
    st.title("🤖 المساعد الذكي")
    st.caption("اسألني عن الأرباح، الموردين، أو اطلب اقتراح سعر")

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    user_input = st.chat_input("اكتب سؤالك هنا...")

    if user_input:
        st.session_state.history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        intent     = detect_intent(user_input)
        confidence = get_confidence(user_input)
        top_conf   = list(confidence.values())[0]

        response = ""

        if intent == "client_profit":
            client_name = extract_name_after_keyword(user_input, "العميل")
            if client_name:
                result = merged[merged["client_name"].str.lower() == client_name.lower()]
                if not result.empty:
                    total = result["profit"].sum()
                    response = f"💰 إجمالي الربح من **{client_name}**: {fmt_currency(total)}\n\n"
                    response += f"عدد المعاملات: {len(result)} — متوسط الهامش: {fmt_percent(result['margin'].mean())}"
                else:
                    response = f"❌ لم أجد العميل '{client_name}'"
            else:
                report = client_profit_report(merged)
                lines  = [f"- **{r['client_name']}**: {fmt_currency(r['total_profit'])}" for _, r in report.iterrows()]
                response = "📊 أرباح جميع العملاء:\n\n" + "\n".join(lines)

        elif intent == "supplier_check":
            changes = all_supplier_changes(purchases)
            if changes.empty:
                response = "لا توجد بيانات كافية للمقارنة."
            else:
                changes = changes.merge(products[["product_id","product_name"]], on="product_id", how="left")
                lines   = []
                for _, r in changes.iterrows():
                    icon = "🔴" if r["direction"]=="up" else "🟢" if r["direction"]=="down" else "🟡"
                    lines.append(f"{icon} {r.get('product_name','')}: {r['change_percent']:+.1f}% ({fmt_currency(r['old_price'])} → {fmt_currency(r['new_price'])})")
                response = "📦 تغيرات أسعار الموردين:\n\n" + "\n".join(lines)

        elif intent == "suggest_price":
            response = "🧮 انتقل إلى **حاسبة سعر البيع** في القائمة الجانبية لإدخال تكلفة المنتج والحصول على السعر المقترح."

        elif intent == "low_margin":
            low = merged[merged["margin"] < 15][["product_name","client_name","margin","profit"]]
            if low.empty:
                response = "✅ جميع المنتجات بهامش أعلى من 15%."
            else:
                lines = [f"- {r['product_name']} ({r['client_name']}): هامش {fmt_percent(r['margin'])}" for _, r in low.iterrows()]
                response = f"⚠️ {len(low)} منتج بهامش ضعيف:\n\n" + "\n".join(lines)

        elif intent == "top_products":
            top = merged.groupby("product_name")["profit"].sum().nlargest(5)
            lines = [f"{i+1}. **{p}**: {fmt_currency(v)}" for i, (p, v) in enumerate(top.items())]
            response = "🏆 أفضل 5 منتجات ربحًا:\n\n" + "\n".join(lines)

        elif intent == "total_profit":
            total = merged["profit"].sum()
            response = f"💰 إجمالي الأرباح الكلية: **{fmt_currency(total)}**"

        else:
            response = (
                "🤔 لم أفهم السؤال جيدًا. يمكنني الإجابة عن:\n\n"
                "- أرباح العملاء (مثال: *كم ربحنا من العميل ClientA*)\n"
                "- تغيرات أسعار الموردين\n"
                "- المنتجات ذات الهامش الضعيف\n"
                "- أفضل المنتجات ربحًا\n"
                "- إجمالي الأرباح"
            )

        confidence_note = f"\n\n*ثقة النموذج: {top_conf*100:.0f}%*"
        full_response   = response + confidence_note
        st.session_state.history.append(("assistant", full_response))
        st.chat_message("assistant").write(full_response)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Evaluation":
    st.title("🔬 تقييم نموذج NLP")
    st.caption("Track 1: NLP — TF-IDF + Logistic Regression Intent Classifier")

    c1, c2, c3 = st.columns(3)
    c1.metric("دقة النموذج (Accuracy)", f"{model_metrics['accuracy']}%")
    c2.metric("بيانات التدريب",         f"{model_metrics['n_train']} جملة")
    c3.metric("بيانات الاختبار",         f"{model_metrics['n_test']} جملة")

    st.divider()
    st.subheader("Classification Report")
    st.code(model_metrics["report"], language="text")

    st.divider()
    st.subheader("اختبر النموذج مباشرة")
    test_input = st.text_input("أدخل جملة لاختبار تصنيفها:")
    if test_input:
        intent = detect_intent(test_input)
        conf   = get_confidence(test_input)
        st.success(f"النية المكتشفة: **{intent}**")
        st.bar_chart(conf)

    st.divider()
    st.subheader("المعمارية التقنية للنموذج")
    st.markdown("""
| المرحلة | التقنية | الوصف |
|---|---|---|
| تمثيل النص | TF-IDF (char n-grams 2-4) | يحول الجمل العربية إلى متجهات رقمية |
| التصنيف | Logistic Regression | يتعلم تمييز النوايا من المتجهات |
| التطبيع | Normalization | إزالة علامات الترقيم وتوحيد الحروف |
| التدريب | Train/Test Split 80/20 | تقييم موضوعي على بيانات غير مرئية |
    """)
