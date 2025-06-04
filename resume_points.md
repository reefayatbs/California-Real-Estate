## California Real Estate Price Predictor - Resume Points

• Developed a production-ready REST API using Flask that predicts California real estate prices with 837 cities coverage, handling 100+ requests per minute

• Engineered and deployed an XGBoost model achieving 89% prediction accuracy, outperforming Linear Regression, Random Forest, and Neural Network alternatives through cross-validation

• Implemented a microservices architecture separating model training, API serving, and data preprocessing components for improved maintainability and scalability

• Built robust data preprocessing pipeline handling missing values, outliers, and feature engineering for 150,000+ real estate records

• Designed RESTful endpoints with input validation, rate limiting, and CORS security features for production deployment

• Optimized model hyperparameters using GridSearchCV, reducing average prediction error to $245K on California housing prices ranging from $500K to $3.5M

Technical Stack:
- Backend: Python, Flask, XGBoost, scikit-learn
- Data Processing: pandas, numpy
- API Security: Flask-Limiter, CORS
- Development: Git, Docker 