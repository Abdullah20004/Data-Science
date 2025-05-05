import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tabulate import tabulate

# Set page config
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Title and introduction
st.title("Online Retail Customer Segmentation")
st.markdown("""
Welcome to the Customer Segmentation Dashboard! This app segments customers from the Online Retail dataset using KMeans clustering, answering key questions about purchasing behavior. Explore clusters, visualize insights, and predict segments for new customers.
""")

# Load and preprocess data
@st.cache_recource
def load_data():
    df = pd.read_excel('Online Retail (1).xlsx')
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['Country'].isin(['Unspecified', 'UNKNOWN', 'unknown', '', ' '])]
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Feature engineering
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Frequency'] = df.groupby('CustomerID')['CustomerID'].transform('count')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.month
    df['FirstPurchase'] = df.groupby('CustomerID')['InvoiceDate'].transform('min')
    df['CustomerTenure'] = (df['InvoiceDate'] - df['FirstPurchase']).dt.days
    df['DescriptionFreq'] = df['Description'].map(df['Description'].value_counts())

    # Continent mapping
    country_to_continent = {
        'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe', 'EIRE': 'Europe',
        'Spain': 'Europe', 'Netherlands': 'Europe', 'Belgium': 'Europe', 'Switzerland': 'Europe',
        'Portugal': 'Europe', 'Australia': 'Oceania', 'Norway': 'Europe', 'Italy': 'Europe',
        'Channel Islands': 'Europe', 'Finland': 'Europe', 'Cyprus': 'Europe', 'Sweden': 'Europe',
        'Austria': 'Europe', 'Denmark': 'Europe', 'Japan': 'Asia', 'Poland': 'Europe',
        'Israel': 'Asia', 'USA': 'North America', 'Hong Kong': 'Asia', 'Singapore': 'Asia',
        'Iceland': 'Europe', 'Canada': 'North America', 'Greece': 'Europe', 'Malta': 'Europe',
        'United Arab Emirates': 'Asia', 'European Community': 'Europe', 'RSA': 'Africa',
        'Lebanon': 'Asia', 'Lithuania': 'Europe', 'Brazil': 'South America',
        'Czech Republic': 'Europe', 'Bahrain': 'Asia', 'Saudi Arabia': 'Asia'
    }
    df['Continent'] = df['Country'].map(country_to_continent)

    # Remove outliers
    def remove_outliers_iqr(df, columns):
        df_clean = df.copy()
        for col in columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean

    df = remove_outliers_iqr(df, columns=['Quantity', 'UnitPrice'])
    return df

# Load data
df = load_data()
df_original = df.copy()

# Preprocessing
numeric_columns = ['TotalPrice', 'Frequency', 'Month', 'CustomerTenure', 'DescriptionFreq']
scaler = StandardScaler()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_encoded = ohe.fit_transform(df[['Continent']])
ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(['Continent']))
df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1).drop(columns=['Continent'])
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Features
features = numeric_columns + [col for col in df.columns if col.startswith('Continent_')]
X = df[['CustomerID'] + features].copy()

# Split data
X_temp, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.25, random_state=42)
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# PCA and KMeans
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train[features])
X_val_pca = pca.transform(X_val[features])
X_test_pca = pca.transform(X_test[features])
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_train_pca)

# Save models
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Create train_df and cluster_profiles
df_original_agg = df_original.groupby('CustomerID').agg({
    'TotalPrice': 'mean',
    'Frequency': 'mean',
    'Month': lambda x: x.mode()[0] if not x.mode().empty else None,
    'CustomerTenure': 'max',
    'Description': lambda x: x.mode()[0] if not x.mode().empty else None,
    'Country': lambda x: x.mode()[0] if not x.mode().empty else None,
    'DescriptionFreq': 'mean'
}).reset_index()

train_df = X_train[['CustomerID']].copy()
train_df['Cluster'] = kmeans.predict(X_train_pca)
train_df = train_df.merge(df_original_agg, on='CustomerID', how='left')

def mode_series(series):
    return series.mode()[0] if not series.mode().empty else None

cluster_profiles = train_df.groupby('Cluster').agg({
    'TotalPrice': 'mean',
    'Frequency': 'mean',
    'CustomerTenure': 'mean',
    'Month': mode_series,
    'Description': mode_series,
    'Country': mode_series,
    'DescriptionFreq': 'mean',
    'CustomerID': 'count'
}).reset_index()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Cluster Analysis", "Research Questions", "Predict Customer", "About"])

if page == "Home":
    st.header("Project Overview")
    st.markdown("""
    **Objective**: Segment customers in the Online Retail dataset to understand purchasing behavior.
    **Dataset**: Transactions with features like `CustomerID`, `TotalPrice`, `Frequency`, `CustomerTenure`, etc.
    **Model**: KMeans clustering with 5 clusters, optimized using the elbow method.
    **Research Questions**:
    - Q1: What are the key factors that influence customer purchasing behavior?
    - Q2: How does customer purchase frequency correlate with order value?
    - Q3: Which individual products have the highest number of returns?
    - Q4: Is there a seasonal trend in online retail sales, and how does it impact revenue?
    - Q5: How does customer behavior vary across different countries?
    """)

elif page == "Cluster Analysis":
    st.header("Cluster Analysis")
    
    # Cluster profiles
    st.subheader("Cluster Profiles")
    st.write(cluster_profiles)
    
    # Interactive cluster plot
    st.subheader("Cluster Visualization")
    x_axis = st.selectbox("X-axis feature", ['PCA0', 'Frequency', 'TotalPrice', 'CustomerTenure'])
    y_axis = st.selectbox("Y-axis feature", ['PCA1', 'TotalPrice', 'Frequency', 'CustomerTenure'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if x_axis == 'PCA0' and y_axis == 'PCA1':
        plot_df = pd.DataFrame(X_train_pca, columns=[f'PCA{i}' for i in range(X_train_pca.shape[1])])
        plot_df['Cluster'] = train_df['Cluster']
        sns.scatterplot(data=plot_df, x='PCA0', y='PCA1', hue='Cluster', palette='tab10', ax=ax)
    else:
        sns.scatterplot(data=train_df, x=x_axis, y=y_axis, hue='Cluster', palette='tab10', ax=ax)
    plt.title(f'{x_axis} vs {y_axis} by Cluster')
    st.pyplot(fig)

elif page == "Research Questions":
    st.header("Research Questions")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Q1", "Q2", "Q3", "Q4", "Q5"])
    
    with tab1:
        st.subheader("Q1: Key factors in purchasing behavior")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=train_df, x='Frequency', y='TotalPrice', hue='Cluster', palette='tab10', ax=ax)
        plt.title('Customer Purchase Frequency vs Total Price by Cluster')
        plt.xlabel('Purchase Frequency')
        plt.ylabel('Total Price')
        st.pyplot(fig)
        st.markdown("**Insight**: High-frequency, high-price clusters indicate loyal spenders; low-frequency, high-price suggest occasional big purchases.")
    
    with tab2:
        st.subheader("Q2: Purchase frequency vs order value")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=train_df, x='Cluster', y='TotalPrice', ax=ax)
        plt.title('Total Price Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Total Price')
        st.pyplot(fig)
        st.markdown("**Insight**: Higher-frequency clusters often have higher median total price, indicating a positive correlation.")
    
    with tab3:
        st.subheader("Q3: Product return rates")
        df_original['IsReturn'] = df_original['Quantity'] < 0
        top_returned_products = df_original[df_original['IsReturn']].groupby('Description').size().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_returned_products.plot(kind='bar', color='red', ax=ax)
        plt.title('Top 10 Returned Products by Return Count')
        plt.xlabel('Product Description')
        plt.ylabel('Number of Returns')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("**Insight**: These products have the highest returns, possibly due to quality issues, popularity, or shipping errors.")
    
    with tab4:
        st.subheader("Q4: Seasonal trends")
        monthly_sales = df_original.groupby('Month')['TotalPrice'].sum()
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_sales.plot(kind='line', marker='o', color='purple', ax=ax)
        plt.title('Total Sales by Month')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.xticks(range(1, 13))
        plt.grid(True)
        st.pyplot(fig)
        st.markdown("**Insight**: Sales peak in November, likely driven by holiday shopping.")
    
    with tab5:
        st.subheader("Q5: Behavior by country")
        country_spending = train_df.groupby('Country')['TotalPrice'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        country_spending.plot(kind='bar', color='teal', ax=ax)
        plt.title('Average Spending per Transaction by Country')
        plt.xlabel('Country')
        plt.ylabel('Average Total Price')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("**Insight**: The UK dominates spending, with Australia and Netherlands showing high per-transaction values.")

elif page == "Predict Customer":
    st.header("Predict Customer Segment")
    
    st.markdown("Enter customer details to predict their segment and see top products:")
    total_price = st.number_input("Total Price (e.g., 100)", min_value=0.0, value=100.0)
    frequency = st.number_input("Purchase Frequency (e.g., 10)", min_value=0.0, value=10.0)
    month = st.slider("Month", 1, 12, 6)
    customer_tenure = st.number_input("Customer Tenure (days, e.g., 100)", min_value=0.0, value=100.0)
    description_freq = st.number_input("Description Frequency (e.g., 50)", min_value=0.0, value=50.0)
    continent = st.selectbox("Continent", ['Europe', 'Asia', 'North America', 'Oceania', 'Africa', 'South America'])
    
    if st.button("Predict"):
        # Load models
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('ohe.pkl', 'rb') as f:
            ohe = pickle.load(f)
        with open('pca.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('kmeans.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        
        # Prepare input
        input_data = pd.DataFrame({
            'TotalPrice': [total_price],
            'Frequency': [frequency],
            'Month': [month],
            'CustomerTenure': [customer_tenure],
            'DescriptionFreq': [description_freq],
            'Continent': [continent]
        })
        
        # Encode and scale
        ohe_encoded = ohe.transform(input_data[['Continent']])
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(['Continent']))
        input_data = pd.concat([input_data.drop(columns=['Continent']), ohe_df], axis=1)
        input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])
        
        # Ensure all feature columns are present
        for col in features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[features]
        
        # PCA and predict
        input_pca = pca.transform(input_data)
        cluster = kmeans.predict(input_pca)[0]
        
        st.success(f"Predicted Cluster: {cluster}")
        st.markdown(f"**Cluster Characteristics**:")
        st.dataframe(cluster_profiles[cluster_profiles['Cluster'] == cluster])
        
        # Top products for the cluster
        df_original['Cluster'] = kmeans.predict(pca.transform(X[features]))
        cluster_products = df_original[df_original['Cluster'] == cluster]['Description']
        top_products = cluster_products.value_counts().head(3).index.tolist()
        
        st.markdown(f"**Top 3 Products for Cluster {cluster}**:")
        for i, product in enumerate(top_products, 1):
            st.write(f"{i}. {product}")

elif page == "About":
    st.header("About")
    st.markdown("""
    **Student**: Abdalla Ahmed Elsherbiny  
    **Email**: AE2401530@tkh.edu  
    **University ID**: 2401530  
    **Role**: Data Science Student  
    **Institution**: Coventry University  
    **Tools Used**: Python, Pandas, Scikit-learn, Streamlit, Matplotlib, Seaborn  
    **Contact**: 01020493955  
    This project segments customers to inform retail strategies, answering key questions about behavior and trends.
    """)
