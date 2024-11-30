# stockprice

## Live Demo

The **Stock Market Predictor** application is deployed and can be accessed online. You can explore the application and try out the stock prediction feature through the following link:
[Stock Price Predictor - Live Demo](https://stockprice-6enjzsjla9nooqfybejgtv.streamlit.app/)

[Stock Price Predictor - Live Demo](https://stockpricest125050-hfa8h2hae7cqeeay.canadacentral-01.azurewebsites.net)

Feel free to select a stock ticker, choose the prediction timeframe, and get predictions for the stock price!

---

This will make it clear to users where they can access the live version of the app, and the link will be easy to click for anyone visiting your repository.
## Dataset Explanation

The dataset used in this project contains historical stock data, including daily price and volume information for different stocks. Below is an explanation of each column in the dataset:

### Dataset Columns

1. **Date**:
   - Represents the **date** on which the stock data was recorded.
   - The date is displayed in the format `YYYY-MM-DD`.
   - Example: `1980-12-12` (the data corresponds to December 12, 1980).

2. **Open**:
   - The **opening price** of the stock on the given date.
   - This price reflects the value at which the stock started trading on the specific day.
   - Example: `0.128348` (the stock opened at this price on `1980-12-12`).

3. **High**:
   - The **highest price** the stock reached during the trading day.
   - This reflects the peak value of the stock between the market open and close on the given date.
   - Example: `0.128906` (the highest price reached on `1980-12-12`).

4. **Low**:
   - The **lowest price** the stock reached during the trading day.
   - This reflects the lowest value the stock traded at during that day's session.
   - Example: `0.128348` (the lowest price on `1980-12-12`).

5. **Close**:
   - The **closing price** of the stock when the market closed on a particular day.
   - This is considered the most important price as it reflects the final transaction of the day.
   - Example: `0.128348` (the closing price on `1980-12-12`).

6. **Adj Close** (Adjusted Close):
   - The **adjusted closing price** after accounting for corporate actions like stock splits, dividends, and new stock issues.
   - It provides a more accurate reflection of the stock's value, especially when comparing stock performance over time.
   - Example: `0.100323` (the adjusted close on `1980-12-12`, which could reflect adjustments due to corporate actions like a stock split or dividend payouts).

7. **Volume**:
   - The **number of shares** that were traded on that day.
   - This represents the total trading activity for the stock on the given day.
   - Example: `469033600` (the trading volume on `1980-12-12`).

8. **Stock**:
   - The **ticker symbol** of the stock being tracked.
   - In this dataset, all rows have the ticker symbol `AAPL`, which refers to **Apple Inc.**.
   - Example: `AAPL` (Apple Inc. stock).

---

### Example Row Breakdown

Here is an example of a row from the dataset:

| Date       | Open     | High     | Low      | Close    | Adj Close | Volume     | Stock |
|------------|----------|----------|----------|----------|-----------|------------|-------|
| 1980-12-12 | 0.128348 | 0.128906 | 0.128348 | 0.128348 | 0.100323  | 469033600 | AAPL  |

- **Date**: `1980-12-12`
- **Open**: `0.128348` (The stock opened at this price)
- **High**: `0.128906` (The highest price reached during the day)
- **Low**: `0.128348` (The lowest price reached during the day)
- **Close**: `0.128348` (The price at which the stock closed)
- **Adj Close**: `0.100323` (The adjusted close price after corporate actions)
- **Volume**: `469033600` (The number of shares traded during the day)
- **Stock**: `AAPL` (Apple stock)

---

## How It Works

This dataset allows you to analyze stock performance, track price movements, and calculate various metrics such as moving averages, volatility, and trading volume patterns. The data is used to train a machine learning model, specifically an LSTM (Long Short-Term Memory) model, to predict future stock prices.

---

# Stock Market Predictor

This project is a **Stock Market Price Prediction** application using **LSTM (Long Short-Term Memory)** model, built with Python and Streamlit. The app allows users to select a stock ticker, specify a prediction timeframe, and generate predictions of stock prices. It utilizes historical stock data from Yahoo Finance, processes it with LSTM, and plots the predicted and actual stock prices along with other insights like moving averages.

## Features

- **Stock Prediction**: Users can select a stock symbol and specify a time frame (1 Day, 1 Week, 1 Month, 1 Year) for price predictions.
- **Visualizations**: Displays stock price predictions against actual prices, moving averages, and future predictions.
- **Model**: The app uses a deep learning model (LSTM) to predict the future prices based on historical data.
- **Moving Averages**: Plots 50, 100, and 200 day moving averages.

## Requirements

To run this application, ensure you have the following libraries installed:

- Python 3.6 or higher
- Streamlit
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Keras
- TensorFlow
- yfinance

To install the required libraries, you can use the following command:

```bash
pip install streamlit numpy pandas matplotlib scikit-learn keras tensorflow yfinance
```

## How to Run

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/stock-market-predictor.git
cd stock-market-predictor
```

### 2. Run the Streamlit Application

Once all the dependencies are installed, run the app using Streamlit:

```bash
streamlit run app.py
```

This will start a local server, and you can access the app by navigating to the URL shown in the terminal (usually http://localhost:8501).

### 3. Interact with the App

- **Stock Selection**: From the dropdown menu, select the stock ticker you want to predict. The app uses Yahoo Finance to fetch stock data.
- **Timeframe Selection**: Choose a prediction timeframe (1 Day, 1 Week, 1 Month, 1 Year).
- **Prediction**: Click the "Predict" button to generate predictions. The app will display the stock's actual vs predicted prices, future predictions, and moving averages for better insights.

## How It Works

1. **Data Collection**: 
   - Stock data is fetched using `yfinance` from Yahoo Finance for the past 5 years.
   - The stock’s closing price is used to train the model.

2. **Data Preprocessing**:
   - The data is scaled using `MinMaxScaler` to normalize the prices between 0 and 1.
   - The dataset is split into training and testing sets, where 80% of the data is used for training, and the rest is used for testing.

3. **Model Training**:
   - A deep learning model using **LSTM (Long Short-Term Memory)** is built with Keras.
   - The model is trained for 5 epochs using the training data.

4. **Prediction**:
   - After training, the model predicts future stock prices based on the test data.
   - The predictions are then transformed back to their original scale using the inverse transformation of `MinMaxScaler`.

5. **Visualization**:
   - **Actual vs Predicted Price**: A graph comparing the actual stock prices with the predicted prices for the test period.
   - **Future Predictions**: A graph that shows the predicted stock prices for the selected future timeframe.
   - **Moving Averages**: Plots 50, 100, and 200 day moving averages along with the actual closing price to identify trends.

## Example Output

- **Predicted Stock Price**: The predicted stock price for the selected stock at the end of the selected prediction period (e.g., 1 day, 1 week, etc.).
- **Mean Absolute Percentage Error (MAPE)**: The MAPE value indicates how accurate the model's predictions are. A lower MAPE indicates better performance.
- **Root Mean Squared Error (RMSE)**: Another metric to measure the performance of the model.

## Limitations

- **Data Quality**: The model relies on historical data and may not account for sudden market changes, such as major news events or financial reports.
- **Short Timeframes**: Predictions for shorter timeframes may be less accurate due to the nature of stock market volatility.
- **Model Complexity**: The LSTM model used here is relatively simple and might not capture all the complexities of stock market behavior. Further improvements can include adding more features (like technical indicators) or using more sophisticated models.

## Future Improvements

- Implement additional features like technical indicators (e.g., RSI, MACD) for better predictions.
- Use more complex models like **GRU** (Gated Recurrent Unit) or **Transformer** models.
- Enhance the user interface with more interactive charts and options.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Yahoo Finance API**: For providing historical stock market data.
- **Streamlit**: For building the interactive web application.
- **Keras** and **TensorFlow**: For deep learning and model training.


## CI/CD Pipeline and Deployment to Azure Web Apps

This project utilizes **GitHub Actions** to automatically build, test, and deploy the stock market prediction web app to **Azure Web App**. Below is an overview of the GitHub Actions workflow that automates the entire process from building the Docker container to deploying the app on Azure.

### GitHub Actions Workflow

The workflow file (`.github/workflows/azure-web-app.yml`) defines the following steps:

1. **Build Job**:
   - **Checkout the code**: The code is checked out from the GitHub repository using `actions/checkout@v2`.
   - **Setup Docker Buildx**: Docker's Buildx tool is set up for building the container image.
   - **Login to Docker Registry**: Logs in to Docker Hub using credentials stored in GitHub Secrets. This allows the action to push the container image to the Docker registry.
   - **Build and Push Docker Image**: The Docker image is built using the `Dockerfile` in the repository and then pushed to Docker Hub with a tag that includes the GitHub commit SHA.

2. **Deploy Job**:
   - **Deploy to Azure Web App**: After the build job is complete, the action uses `azure/webapps-deploy@v2` to deploy the Docker container to **Azure Web Apps**. It uses a publish profile stored in GitHub Secrets to authenticate and push the container image to the Azure web app.
   - The web app name is set to `stockpricest125050`, and the deployment happens in the `production` slot.

### Steps in the Deployment Workflow

#### 1. **Build and Push Docker Image**
```yaml
- name: Build and push container image to registry
  uses: docker/build-push-action@v3
  with:
    push: true
    tags: index.docker.io/${{ secrets.AzureAppService_ContainerUsername }}/st125050/mlfinalproject:${{ github.sha }}
    file: ./Dockerfile
```
- The Docker image is built based on the `Dockerfile` in the root directory of the project.
- The image is pushed to Docker Hub using a unique tag that corresponds to the current commit (`${{ github.sha }}`).

#### 2. **Deploy to Azure Web App**
```yaml
- name: Deploy to Azure Web App
  id: deploy-to-webapp
  uses: azure/webapps-deploy@v2
  with:
    app-name: 'stockpricest125050'
    slot-name: 'production'
    publish-profile: ${{ secrets.AzureAppService_PublishProfile }}
    images: 'index.docker.io/${{ secrets.AzureAppService_ContainerUsername }}/st125050/mlfinalproject:${{ github.sha }}'
```
- The Docker image is deployed to the **Azure Web App** named `stockpricest125050` in the `production` slot.
- The deployment is authenticated using a **Publish Profile** stored as a GitHub Secret (`AzureAppService_PublishProfile`).

### GitHub Secrets

To securely manage credentials for deployment, the following **GitHub Secrets** should be set up in the repository:

- **AzureAppService_ContainerUsername**: Your Docker Hub username.
- **AzureAppService_ContainerPassword**: Your Docker Hub password.
- **AzureAppService_PublishProfile**: The publish profile from Azure Web Apps, which contains the credentials needed to deploy the app.

### Triggering the Workflow

The workflow can be triggered in two ways:

1. **Push to the `main` branch**: Every time there is a push to the `main` branch, this workflow is triggered, automatically building and deploying the latest changes.
   
   ```yaml
   on:
     push:
       branches:
         - main
   ```

2. **Manual Trigger (Workflow Dispatch)**: The workflow can also be manually triggered from the GitHub Actions interface.

   ```yaml
   on:
     workflow_dispatch:
   ```

### Benefits of This Setup

- **Automated Deployment**: Every change to the `main` branch is automatically built and deployed to Azure Web Apps, ensuring that the latest changes are always live.
- **Containerization**: Using Docker ensures that the app runs in a consistent environment across different stages (development, testing, production).
- **Scalable and Reliable**: Azure Web Apps provide scalability, so the app can handle increasing traffic without requiring manual intervention.

### Troubleshooting

If there are any issues with the deployment process, you can check the **GitHub Actions** logs for detailed error messages. Common issues could include:

- Incorrect credentials stored in GitHub Secrets.
- Errors in the Dockerfile causing the build to fail.
- Incorrect Azure Web App settings or insufficient permissions.

## Conclusion

This GitHub Actions setup automates the process of building and deploying the stock prediction web app to **Azure Web App**, ensuring efficient and reliable updates. The CI/CD pipeline is triggered on every change to the `main` branch, allowing for continuous integration and deployment with minimal manual intervention.

---

This section will give users a complete overview of how the CI/CD process works, from building and pushing Docker containers to deploying the app to Azure Web Apps using GitHub Actions.
