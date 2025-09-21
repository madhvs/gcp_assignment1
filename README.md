# gcp_assignment1
# Stock News Analysis Pipeline

An intelligent pipeline that extracts stock tickers from company names, fetches recent news, and performs sentiment analysis using Google Gemini AI with comprehensive MLflow tracking and tracing.

## Features

- 🔍 **Automatic Stock Ticker Extraction** from company names using Yahoo Finance API
- 📰 **Real-time News Fetching** using Tavily Search API
- 🤖 **AI-Powered Sentiment Analysis** with Google Gemini 2.0 Flash
- 📊 **Complete MLflow Tracking** with runs, metrics, parameters, and artifacts
- 🔗 **Distributed Tracing** with span-based performance monitoring
- 📈 **Structured JSON Output** with sentiment, entities, and market implications

## Prerequisites

- Python 3.8+
- Google Cloud Project with Vertex AI API enabled
- MLflow tracking server (local or remote)
- Tavily Search API key
- Google Cloud credentials configured

## Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd stock-news-analysis
```

### 2. Create Virtual Environment
```bash
python -m venv stock_analysis_env
source stock_analysis_env/bin/activate  # On Windows: stock_analysis_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with:
```txt
yfinance==0.2.28
requests==2.31.0
langchain==0.1.20
langchain-google-vertexai==1.0.6
google-cloud-aiplatform==1.60.0
mlflow==2.15.1
tavily-python==0.3.3
google-generativeai==0.7.2
```

## Configuration

### 1. Google Cloud Setup

#### Step 1: Enable APIs
```bash
# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com
```

#### Step 2: Set Up Authentication
```bash
# Option 1: Using service account key (recommended for production)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"

# Option 2: Using gcloud CLI (recommended for development)
gcloud auth application-default login
```

#### Step 3: Set Environment Variables
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-central1"  # or your preferred region
```

### 2. Tavily Search API Setup

#### Step 1: Get API Key
1. Visit [Tavily API](https://tavily.com/)
2. Sign up and get your API key
3. Set environment variable:
```bash
export TAVILY_API_KEY="tvly-your-api-key-here"
```

### 3. MLflow Configuration

#### Option A: Local MLflow Server
```bash
# Start local MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Set tracking URI in your code or environment
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

#### Option B: Remote MLflow Server
```bash
# If using a remote MLflow server, update the tracking URI in the code:
# mlflow.set_tracking_uri("http://your-mlflow-server:5000/")
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000/"
```

#### Step 1: Create MLflow Experiment
```bash
# Using MLflow CLI
mlflow experiments create -n "stock_news_analysis_v2"

# Or it will be created automatically when you run the pipeline
```

### 4. Update Configuration in Code

Edit the configuration section in `stock_analysis.py`:

```python
# Update these values according to your setup
PROJECT_ID = "your-google-cloud-project-id"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

# Update MLflow tracking URI
mlflow.set_tracking_uri("http://your-mlflow-server:5000/")
mlflow.set_experiment("your_experiment_name")

# Tavily API key (set in environment or directly)
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"
```

## Usage

### Basic Execution
```bash
python stock_analysis.py
```

When prompted, enter a company name:
```
Enter company name: Apple Inc
```

### Sample Commands

#### Example 1: Analyze Apple Inc
```bash
python stock_analysis.py
# Input: Apple Inc
# Expected ticker: AAPL
```

#### Example 2: Analyze Tesla
```bash
python stock_analysis.py
# Input: Tesla
# Expected ticker: TSLA
```

#### Example 3: Analyze Microsoft
```bash
python stock_analysis.py
# Input: Microsoft Corporation
# Expected ticker: MSFT
```

### Expected Output

The pipeline will output:
```
Step 1: Getting ticker symbol...
Apple Inc: AAPL

Step 2: Getting news...
Searching for news about the stock: Apple Inc
============================================================
Found 5 news articles

Step 3: Analyzing with Gemini...

Analysis Results:
============================================================
{
  "company_name": "Apple Inc",
  "stock_code": "AAPL",
  "newsdesc": "Recent news about Apple Inc covers...",
  "sentiment": "Positive",
  "people_names": ["Tim Cook", "..."],
  "places_names": ["Cupertino", "..."],
  "other_companies_referred": ["Microsoft", "..."],
  "related_industries": ["Technology", "Consumer Electronics"],
  "market_implications": "The recent developments suggest...",
  "confidence_score": 0.87
}

Pipeline completed. Total time: 23.45 seconds
```

## MLflow Monitoring

### 1. View Experiments
```bash
# Open MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

Navigate to `http://localhost:5000` to view:
- **Experiments**: All pipeline runs with parameters and metrics
- **Traces**: Detailed span-based tracing of pipeline execution
- **Artifacts**: Generated files, prompts, and analysis results

### 2. Key Metrics Tracked
- Pipeline execution time
- Number of news articles found
- Sentiment confidence score
- Entity extraction counts (people, companies, industries)

### 3. Artifacts Stored
- Input prompts
- News articles text
- Function schema
- Analysis output JSON
- Summary and market implications

## Troubleshooting

### Common Issues

#### 1. MLflow UUID Already Active Error
**Solution**: Restart your Python session or run:
```python
import mlflow
if mlflow.active_run():
    mlflow.end_run()
```

#### 2. Google Cloud Authentication Error
**Solution**: 
```bash
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

#### 3. Empty Traces in MLflow
**Solution**: Ensure you're using the updated code with `mlflow.start_span()` and check the "Traces" tab in MLflow UI.

#### 4. Tavily API Rate Limiting
**Solution**: Add delays between requests or upgrade your Tavily plan.

#### 5. No News Found
**Solution**: 
- Check your Tavily API key
- Try different company names
- Verify internet connectivity

### Environment Variables Checklist
```bash
# Required environment variables
echo $GOOGLE_CLOUD_PROJECT
echo $GOOGLE_CLOUD_REGION
echo $TAVILY_API_KEY
echo $MLFLOW_TRACKING_URI
echo $GOOGLE_APPLICATION_CREDENTIALS  # If using service account
```

## Architecture

```
[User Input] → [Ticker Extraction] → [News Fetching] → [Sentiment Analysis] → [Results]
     ↓              ↓                     ↓                  ↓               ↓
[MLflow Tracking] → [Spans] → [Metrics] → [Artifacts] → [Trace Visualization]
```

## API Rate Limits

- **Yahoo Finance**: No official rate limits, but use responsibly
- **Tavily Search**: Check your plan limits (typically 1000 requests/month for free)
- **Google Gemini**: 15 RPM for free tier, higher for paid plans

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review MLflow logs and traces
3. Verify all API keys and credentials
4. Create an issue in the repository

---

**Note**: This pipeline requires active internet connection for API calls to Yahoo Finance, Tavily, and Google Gemini services.
