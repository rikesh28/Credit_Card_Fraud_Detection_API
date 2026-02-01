# Fraud Detection API
A REST API that predicts fraud in real-time. Built with FastAPI and deployed on Render.

🌐 **Live API:** https://credit-card-fraud-detection-api-lmas.onrender.com  
📚 **API Docs:** https://credit-card-fraud-detection-api-lmas.onrender.com/docs

## What It Does

Send it a transaction, get back whether it's fraud or not. Simple as that.

It runs an XGBoost model trained on 590K real e-commerce transactions and gives you:
- Fraud probability (0-100%)
- Risk level (Low/Medium/High/Critical)
- Recommendation (Approve/Review/Decline)

## Try It Out

**Check if it's alive:**
```bash
curl https://credit-card-fraud-detection-api-lmas.onrender.com/
```

**Predict on a transaction:**
```bash
curl -X POST "https://credit-card-fraud-detection-api-lmas.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 150.50,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
```

Should return something like:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.23,
  "risk_level": "Low",
  "message": "Transaction appears legitimate"
}
```

## Endpoints

**GET /** - Health check  
**POST /predict** - Single transaction  
**POST /predict/batch** - Upload CSV, get predictions for all rows  
**GET /model/info** - Model details

Check out `/docs` for the full interactive documentation.

## How I Built This

Started with a 434-feature model that was super accurate but way too slow for production. Had to make some tough choices:

**The Problem:**
- Original model needed 500ms per prediction
- Required historical transaction data (not available in real-time)
- Would've needed an expensive data warehouse

**My Solution:**
- Analyzed which features actually mattered
- Cut down to 21 features that are available immediately
- Retrained model - dropped from 88% to 82% accuracy
- But now it runs in under 100ms

Turns out that's good enough and way more useful.

## Running Locally
```bash
git clone [https://github.com/rikesh28/Credit_Card_Fraud_Detection_API]
cd Credit_Card_Fraud_Detection_API

# Install
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload
```

API will be at `http://localhost:8000`

## Tech Stack

- **FastAPI** - Because it's fast and has built-in docs
- **XGBoost** - Best performance on my data
- **Pydantic** - Input validation
- **Render** - Free hosting (cold starts after 15min though)

## Performance

- **Latency:** <100ms (after cold start)
- **Accuracy:** 82% ROC-AUC
- **Recall:** 67% (catches 2 out of 3 frauds)
- **Precision:** 12% (false alarm rate is acceptable)

Not perfect, but good enough to be useful in production.

## Deployment

Deployed on Render's free tier. First request after inactivity takes 30-60 seconds (cold start), then it's fast.

If I was doing this for real, I'd:
- Use a paid tier to avoid cold starts
- Add monitoring and logging
- Set up auto-scaling
- Add API authentication

But for a portfolio project, this works great.

## What I Learned

- FastAPI is way easier than Flask for APIs
- Deployment is harder than I thought (Python versions, dependency conflicts, etc.)
- The 80/20 rule is real - 21 features gives you 90% of the performance
- Production constraints matter as much as model accuracy

## Issues?

The model file is kinda big (can't push to GitHub). If you clone this, you'll need to:
1. Train the model using the notebooks repo
2. Or contact me for the model file

---

Built this to learn about deploying ML models. Turns out there's a lot more to it than just training the model.

## Contact
Built by Rikesh Sapkota

- LinkedIn: [https://www.linkedin.com/in/rikesh-sapkota-b0591a29a/]
- Email: rikeshsapkota123@gmail.com