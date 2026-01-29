import requests

url = "http://localhost:8000/predict"

# Test legitimate transaction
legitimate = {
    "TransactionAmt": 45.50,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 150.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "debit",
    "addr1": 315.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
}

response = requests.post(url, json=legitimate)
result = response.json()

print("=== LEGITIMATE TRANSACTION ===")
print(f"Fraud Probability: {result['fraud_probability']*100:.2f}%")
print(f"Is Fraud: {result['is_fraud']}")
print(f"Risk Level: {result['risk_level']}")

