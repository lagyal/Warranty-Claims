<h3>Warranty Claim Prediction</h3>

<p>This project focuses on predicting warranty claim fraud using the Kaggle <b>Warranty Claims</b> dataset. The objective is to identify potential fraudulent claims based on product, customer, and claim-related features. Insights from this project can help organizations improve warranty management, reduce fraud, and optimize service operations.</p>

<h3>Dataset</h3>
<p>The dataset consists of two parts: <b>train</b> and <b>test</b>.</p>
<li><b>Train dataset:</b> Contains historical claim data with the target column <i>fraud</i>.</li>
<li><b>Test dataset:</b> Contains claim records without the target column <i>fraud</i>.</li>
<p>Both datasets include features such as <i>product_type, purchase_from, claim_value, product_age</i>, and others.</p>

<h3>Data Preprocessing</h3>
<p>The following steps were applied to clean and prepare the data:</p>
<li>Dropped the first unnamed index column from both train and test datasets.</li>
<li>Checked for null values. <i>claim_value</i> had 240 missing in train and 93 in test; imputed using median.</li>
<li>Boxplots revealed outliers in <i>claim_value</i>.</li>
<li>Standardized duplicate entries (e.g., UP → Uttar Pradesh, claim → Claim).</li>
<li>Feature engineering:
    <ul>
        <li>Combined three AC columns into <i>total_AC_issues</i>.</li>
        <li>Combined three TV columns into <i>total_TV_issues</i>.</li>
        <li>Dropped original AC and TV columns.</li>
    </ul>
</li>
<li>Combined train and test datasets and added a new column <i>data</i> (train/test). Assigned placeholder <i>fraud=0</i> for test data.</li>
<li>Dropped <i>car_details</i> due to high correlation with <i>product_age</i>.</li>
<li>Split combined dataset back into <i>train_clean</i> and <i>test_clean</i>.</li>

<h3>Handling Class Imbalance</h3>
<p>The target variable <i>fraud</i> was highly imbalanced. SMOTE (Synthetic Minority Oversampling Technique) was used to oversample the minority class, ensuring balanced training data.</p>

<h3>Model Building</h3>
<p>Several models were trained and evaluated:</p>
<li>Logistic Regression</li>
<li>Random Forest Classifier</li>
<li>XGBoost Classifier</li>
<li>K-Nearest Neighbors (KNN)</li>
<li>Support Vector Classifier (SVC)</li>

<p>SVC and KNN achieved the best performance, and KNN was selected as the final model.</p>

<h3>Model Evaluation</h3>
<p><b>Training Accuracy:</b></p>
<pre>
               precision    recall  f1-score   support

           0       0.98      0.96      0.97      6139
           1       0.96      0.98      0.97      6139

    accuracy                           0.97     12278
   macro avg       0.97      0.97      0.97     12278
weighted avg       0.97      0.97      0.97     12278
</pre>

<p><b>Testing Accuracy:</b></p>
<pre>
               precision    recall  f1-score   support

           0       1.00      0.96      0.98      1536
           1       0.68      0.97      0.80       133

    accuracy                           0.96      1669
   macro avg       0.84      0.97      0.89      1669
weighted avg       0.97      0.96      0.96      1669
</pre>

<p><b>Confusion Matrices:</b></p>
<p>Training:</p>
<pre>
[[5880  259]
 [ 136 6003]]
</pre>

<p>Testing:</p>
<pre>
[[1476   60]
 [   4  129]]
</pre>

<h3>Feature Importance</h3>
<p>Permutation importance revealed the most influential features:</p>
<li>claim_value</li>
<li>product_age</li>
<li>purpose_complaint</li>
<li>service_center</li>
<li>purchase_from_manufacturer</li>

<p>Claim value and product age were the top predictors of warranty claim fraud.</p>

<h3>Future Work</h3>
<li>Experiment with advanced ensemble models to improve performance.</li>
<li>Implement hyperparameter tuning for KNN and other models.</li>
<li>Feature engineering using domain expertise (e.g., product category, purchase season).</li>
<li>Deploy the model as an API for real-time warranty claim fraud detection.</li>

<h3>Technologies Used</h3>
<li>Python: pandas, numpy, matplotlib, seaborn</li>
<li>Machine Learning: scikit-learn, XGBoost</li>
<li>Data Handling: SMOTE for class balancing</li>
<li>Visualization: Boxplots, heatmaps, ROC curve, feature importance plots</li>
