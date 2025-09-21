import yfinance as yf
import requests
from langchain.tools.tavily_search import TavilySearchResults
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from google import genai

import os
import json
import time
from datetime import datetime

import mlflow
import mlflow.tracking
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

mlflow.set_tracking_uri("http://20.75.92.162:5000/")
mlflow.set_experiment("madhan_stock_news_analysi_v2")

PROJECT_ID = "bdc-trainings"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

from google import genai

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Set up your API keys and project info
os.environ["TAVILY_API_KEY"] = "xxx"

def get_ticker_from_company(company_name, parent_run_id=None):
    # Create explicit child run for tracing with nested=True
    child_run = mlflow.start_run(
        run_name="stock_code_extraction", 
        tags={MLFLOW_PARENT_RUN_ID: parent_run_id} if parent_run_id else {},
        nested=True
    )
    
    try:
        start_time = time.time()
        
        # Log span start
        mlflow.log_param("operation", "stock_code_extraction")
        mlflow.log_param("company_name", company_name)
        mlflow.log_metric("span_start_time", start_time)
        
        # Yahoo Finance search URL
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        mlflow.log_param("search_url", url)
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data['quotes']:
            ticker = data['quotes'][0]['symbol']
            # Log span success
            mlflow.log_param("ticker_found", ticker)
            mlflow.log_param("span_status", "SUCCESS")
            mlflow.log_metric("quotes_count", len(data['quotes']))
            mlflow.log_metric("span_duration_seconds", time.time() - start_time)
            mlflow.log_metric("span_end_time", time.time())
            return ticker
        else:
            # Log span failure
            mlflow.log_param("span_status", "FAILED")
            mlflow.log_param("failure_reason", "no_quotes_found")
            mlflow.log_metric("span_duration_seconds", time.time() - start_time)
            mlflow.log_metric("span_end_time", time.time())
            return None
            
    except Exception as e:
        # Log span error
        mlflow.log_param("span_status", "ERROR")
        mlflow.log_param("error_message", str(e))
        mlflow.log_metric("span_duration_seconds", time.time() - start_time)
        mlflow.log_metric("span_end_time", time.time())
        print(f"Error: {e}")
        return None
    finally:
        # Always end the child run
        mlflow.end_run()

def get_company_news(company_name: str, max_results: int = 5, parent_run_id=None):
    # Create explicit child run for tracing with nested=True
    child_run = mlflow.start_run(
        run_name="news_fetching", 
        tags={MLFLOW_PARENT_RUN_ID: parent_run_id} if parent_run_id else {},
        nested=True
    )
    
    try:
        start_time = time.time()
        
        # Log span start
        mlflow.log_param("operation", "news_fetching")
        mlflow.log_param("company_name", company_name)
        mlflow.log_param("max_results", max_results)
        mlflow.log_metric("span_start_time", start_time)
        
        search_tool = TavilySearchResults(
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            include_raw_content=True
        )
        
        search_query = f"{company_name} latest news "
        mlflow.log_param("search_query", search_query)
        
        print(f"Searching for news about the stock: {company_name}")
        results = search_tool.run(search_query)
        news = []
        print("=" * 60)
        for i, result in enumerate(results, 1):
            news_content = result.get('content', 'No content')[:300]
            news.append(news_content)
            # Log each news item as artifact
            mlflow.log_text(news_content, f"news_article_{i}.txt")
        
        # Log span success
        mlflow.log_param("span_status", "SUCCESS")
        mlflow.log_metric("news_articles_found", len(news))
        mlflow.log_metric("span_duration_seconds", time.time() - start_time)
        mlflow.log_metric("span_end_time", time.time())
        return news
        
    except Exception as e:
        # Log span error
        mlflow.log_param("span_status", "ERROR")
        mlflow.log_param("error_message", str(e))
        mlflow.log_metric("span_duration_seconds", time.time() - start_time)
        mlflow.log_metric("span_end_time", time.time())
        print(f"Error searching for news: {e}")
        return []
    finally:
        # Always end the child run
        mlflow.end_run()

def analyze_news_with_gemini(company_name: str, stock_code: str, news_list: list, parent_run_id=None):
    """
    Use Google GenAI client with function calling to analyze news and return structured JSON
    """
    # Create explicit child run for tracing with nested=True
    child_run = mlflow.start_run(
        run_name="sentiment_parsing", 
        tags={MLFLOW_PARENT_RUN_ID: parent_run_id} if parent_run_id else {},
        nested=True
    )
    
    try:
        start_time = time.time()
        
        # Log span start
        mlflow.log_param("operation", "sentiment_parsing")
        mlflow.log_param("company_name", company_name)
        mlflow.log_param("stock_code", stock_code)
        mlflow.log_param("news_articles_count", len(news_list))
        mlflow.log_param("model", "gemini-2.0-flash")
        mlflow.log_param("temperature", 0.1)
        mlflow.log_metric("span_start_time", start_time)
        
        # Function schema for structured output
        function_schema = {
            "name": "aggregate_company_news",
            "description": "Aggregate multiple news headlines about a company into one JSON object summarizing sentiment, entities, industries, and market implications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": { "type": "string", "description": "Full legal name of the company being analyzed." },
                    "stock_code": { "type": "string", "description": "The company's stock ticker symbol." },
                    "newsdesc": { "type": "string", "description": "One synthesized summary paragraph combining all the provided headlines." },
                    "sentiment": {
                        "type": "string",
                        "enum": ["Positive", "Negative", "Neutral"],
                        "description": "Overall sentiment of the aggregated news."
                    },
                    "people_names": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of individual people mentioned in the news."
                    },
                    "places_names": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of geographic locations mentioned in the news."
                    },
                    "other_companies_referred": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of other companies mentioned in the news besides the main one."
                    },
                    "related_industries": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Industries relevant to the news."
                    },
                    "market_implications": {
                        "type": "string",
                        "description": "Implications of the news on market performance or company stock."
                    },
                    "confidence_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence level for the analysis."
                    }
                },
                "required": [
                    "company_name",
                    "stock_code",
                    "newsdesc",
                    "sentiment",
                    "confidence_score"
                ]
            }
        }
        
        # Initialize the Google GenAI client
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        
        # Prepare news text
        news_text = "\n\n".join([f"News {i+1}: {news}" for i, news in enumerate(news_list)])
        
        # Create the prompt
        prompt = f"""
        Analyze the following news articles about {company_name} (Stock: {stock_code}) and call the aggregate_company_news function with the extracted information.

        News Articles:
        {news_text}

        Instructions:
        1. Synthesize all news into one coherent summary for 'newsdesc'
        2. Determine overall sentiment (Positive/Negative/Neutral)
        3. Extract all person names, places, other companies, and industries mentioned
        4. Analyze market implications
        5. Provide confidence score between 0.0 and 1.0
        
        Use the aggregate_company_news function to return the structured analysis.
        """
        
        # Log prompt and inputs as artifacts
        mlflow.log_text(prompt, "input_prompt.txt")
        mlflow.log_text(news_text, "input_news_articles.txt")
        mlflow.log_dict(function_schema, "function_schema.json")
        
        # Generate content with function calling
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'tools': [{'function_declarations': [function_schema]}],
                'temperature': 0.1
            }
        )
        
        # Extract function call result
        result = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    if part.function_call.name == 'aggregate_company_news':
                        # Convert function call arguments to dict
                        args_dict = {}
                        for key, value in part.function_call.args.items():
                            args_dict[key] = value
                        result = args_dict
                        break
        
        if result:
            # Log span success
            mlflow.log_param("span_status", "SUCCESS")
            mlflow.log_param("sentiment_detected", result.get('sentiment', 'Unknown'))
            mlflow.log_metric("confidence_score", result.get('confidence_score', 0.0))
            mlflow.log_metric("people_count", len(result.get('people_names', [])))
            mlflow.log_metric("companies_count", len(result.get('other_companies_referred', [])))
            mlflow.log_metric("industries_count", len(result.get('related_industries', [])))
            mlflow.log_metric("span_duration_seconds", time.time() - start_time)
            mlflow.log_metric("span_end_time", time.time())
            
            # Log output artifacts
            mlflow.log_dict(result, "analysis_output.json")
            mlflow.log_text(result.get('newsdesc', ''), "summary.txt")
            mlflow.log_text(result.get('market_implications', ''), "market_implications.txt")
            
            return result
        else:
            # Log span failure
            mlflow.log_param("span_status", "FAILED")
            mlflow.log_param("failure_reason", "no_function_call_found")
            mlflow.log_metric("span_duration_seconds", time.time() - start_time)
            mlflow.log_metric("span_end_time", time.time())
            print("No function call found in response")
            return None
        
    except Exception as e:
        # Log span error
        mlflow.log_param("span_status", "ERROR")
        mlflow.log_param("error_message", str(e))
        mlflow.log_metric("span_duration_seconds", time.time() - start_time)
        mlflow.log_metric("span_end_time", time.time())
        print(f"Error in GenAI client execution: {e}")
        return None
    finally:
        # Always end the child run
        mlflow.end_run()

def main():
    # Ensure no runs are active before starting - cleanup any orphaned runs
    if mlflow.active_run():
        print("Warning: Found active MLflow run, ending it before starting new pipeline")
        mlflow.end_run()
    
    # Start main MLflow run
    with mlflow.start_run(run_name=f"stock_analysis_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as parent_run:
        parent_run_id = parent_run.info.run_id
        start_time = time.time()
        
        try:
            # Get company name from user
            company = input("Enter company name: ")
            
            # Log main pipeline parameters
            mlflow.log_param("input_company", company)
            mlflow.log_param("pipeline_version", "1.0")
            mlflow.log_param("timestamp", datetime.now().isoformat())
            mlflow.log_param("mlflow_version", mlflow.__version__)
            
            # Set pipeline tags
            mlflow.set_tag("pipeline_type", "stock_news_analysis")
            mlflow.set_tag("company_input", company)
            
            # Step 1: Get ticker symbol - create child run for tracing
            print("\nStep 1: Getting ticker symbol...")
            ticker = get_ticker_from_company(company, parent_run_id)
            if not ticker:
                # Log pipeline failure
                mlflow.log_param("pipeline_status", "FAILED")
                mlflow.log_param("failure_stage", "ticker_extraction")
                mlflow.log_metric("total_pipeline_time_seconds", time.time() - start_time)
                mlflow.set_tag("pipeline_status", "FAILED")
                print(f"Could not find ticker for {company}")
                return
            
            mlflow.log_param("ticker_extracted", ticker)
            mlflow.set_tag("ticker", ticker)
            print(f"{company}: {ticker}")
            
            # Step 2: Get news - create child run for tracing
            print("\nStep 2: Getting news...")
            news_results = get_company_news(company, max_results=5, parent_run_id=parent_run_id)
            if not news_results:
                # Log pipeline failure
                mlflow.log_param("pipeline_status", "FAILED")
                mlflow.log_param("failure_stage", "news_fetching")
                mlflow.log_metric("total_pipeline_time_seconds", time.time() - start_time)
                mlflow.set_tag("pipeline_status", "FAILED")
                print("No news found")
                return
            
            mlflow.log_metric("news_articles_collected", len(news_results))
            print(f"Found {len(news_results)} news articles")
            
            # Step 3: Analyze with Gemini - create child run for tracing
            print("\nStep 3: Analyzing with Gemini...")
            analysis = analyze_news_with_gemini(company, ticker, news_results, parent_run_id)
            
            if analysis:
                # Log pipeline success
                mlflow.log_param("pipeline_status", "SUCCESS")
                mlflow.log_param("final_sentiment", analysis.get('sentiment', 'Unknown'))
                mlflow.log_metric("final_confidence_score", analysis.get('confidence_score', 0.0))
                mlflow.log_metric("total_pipeline_time_seconds", time.time() - start_time)
                
                # Set final tags for filtering
                mlflow.set_tag("pipeline_status", "SUCCESS")
                mlflow.set_tag("sentiment", analysis.get('sentiment', 'Unknown'))
                
                # Log final output
                mlflow.log_dict(analysis, "final_pipeline_output.json")
                
                print("\nAnalysis Results:")
                print("=" * 60)
                print(json.dumps(analysis, indent=2))
            else:
                # Log pipeline failure
                mlflow.log_param("pipeline_status", "FAILED")
                mlflow.log_param("failure_stage", "sentiment_analysis")
                mlflow.log_metric("total_pipeline_time_seconds", time.time() - start_time)
                mlflow.set_tag("pipeline_status", "FAILED")
                print("Failed to analyze news")
        
        except Exception as e:
            # Log any unexpected errors at pipeline level
            mlflow.log_param("pipeline_status", "ERROR")
            mlflow.log_param("pipeline_error", str(e))
            mlflow.log_metric("total_pipeline_time_seconds", time.time() - start_time)
            mlflow.set_tag("pipeline_status", "ERROR")
            print(f"Pipeline error: {e}")
            raise
        
        finally:
            # Ensure parent run is properly ended
            print(f"\nPipeline completed. Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
