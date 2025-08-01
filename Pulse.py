# dir\utils.py
import streamlit as st
import pandas as pd
import feedparser
import requests
import random
from datetime import datetime
from dateutil import parser
import re
from collections import Counter
from io import BytesIO
import hashlib

# Visualization libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go

# NLP libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

class NewsDataCollector:
	"""Handles news data collection from various RSS sources"""
	
	def __init__(self):
		self.session = requests.Session()
		self.session.headers.update({
			'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
		})
		
	def get_google_news_url(self, query, region='US', language='en', category=None):
		"""Generate Google News RSS URL with parameters"""
		base_url = "https://news.google.com/rss"
		
		if category:
			url = f"{base_url}/headlines/section/topic/{category}?hl={language}&gl={region}&ceid={region}:{language}"
		elif query:
			encoded_query = requests.utils.quote(query)
			url = f"{base_url}/search?q={encoded_query}&hl={language}&gl={region}&ceid={region}:{language}"
		else:
			url = f"{base_url}?hl={language}&gl={region}&ceid={region}:{language}"
			
		return url

	def _create_cache_key(self, url, max_articles):
		"""Create a consistent cache key for RSS requests"""
		return hashlib.md5(f"{url}_{max_articles}".encode()).hexdigest()
	
	@st.cache_data(ttl=300)  # Cache for 5 minutes
	def scrape_rss_feed(_self, url, max_articles=50):
		"""Scrape articles from RSS feed and sort by recency"""
		try:
			feed = feedparser.parse(url)
			articles = []
			
			for entry in feed.entries:
				# Parse the published date for sorting
				published_date = None
				if hasattr(entry, 'published_parsed') and entry.published_parsed:
					try:
						published_date = datetime(*entry.published_parsed[:6])
					except:
						pass
			
				# Fallback: try to parse published string
				if not published_date and entry.get('published'):
					try:
						published_date = parser.parse(entry.published)
					except:
						pass
			
				article = {
					'title': entry.get('title', ''),
					'link': entry.get('link', ''),
					'published': entry.get('published', ''),
					'published_date': published_date,  # Helper field for sorting
					'summary': entry.get('summary', ''),
					'source': entry.get('source', {}).get('title', 'Unknown')
				}
				articles.append(article)
				
			# Sort by published date (most recent first)
			articles.sort(key=lambda x: x['published_date'] or datetime.min, reverse=True)
		
			# Remove helper field and return limited results
			for article in articles:
				del article['published_date']
		
			return articles[:max_articles]

		except Exception as e:
			st.error(f"Error scraping feed: {str(e)}")
			return []
	
	@st.cache_data(ttl=300)  # Cache for 5 minutes
	def collect_news_data(_self, query=None, region='US', category=None, max_articles=50):
		"""Main method to collect news data"""
		url = _self.get_google_news_url(query, region, category=category)
		return _self.scrape_rss_feed(url, max_articles)

class SentimentAnalyzer:
	"""Advanced sentiment analysis using multiple models"""
	

	@st.cache_resource
	def _load_models():
		"""Load sentiment analysis models"""
		try:
			hf_analyzer = pipeline("sentiment-analysis", 
								 model="cardiffnlp/twitter-roberta-base-sentiment-latest")
			return hf_analyzer, True
		except Exception as e:
			st.warning(f"Advanced sentiment model not available, using VADER\n\nReason:\n{e}")
			return SentimentIntensityAnalyzer(), False
	
	def __init__(self):
		self.hf_analyzer, self.use_hf = self._load_models()
		if not self.use_hf:
			self.vader = self.hf_analyzer

	def get_hf_sentiment_label(self, text):
		"""Convert Hugging Face model prediction to descriptive sentiment label"""
		sentiment_class_index = self.hf_analyzer(text)[0]["label"]
		if "0" in sentiment_class_index:
			return "Negative"
		elif "1" in sentiment_class_index:
			return "Neutral"
		else:
			return "Positive"
	
	def get_vader_sentiment_label(self, compound_score):
		"""Convert vader compound score to descriptive label using best practices"""
		if compound_score >= 0.05:
			return "Positive"
		elif compound_score > -0.05:
			return "Neutral"
		else:
			return "Negative"
	
	@st.cache_data
	def analyze_text(_self, text):
		"""Comprehensive sentiment analysis"""
		# Use Hugging Face if available
		if _self.use_hf:
			try:
				return {'sentiment_label': _self.get_hf_sentiment_label(text)}
			except:
				pass
		# Else, use VADER
		return {'sentiment_label': _self.get_vader_sentiment_label(_self.vader.polarity_scores(text)['compound'])}

class TextSummarizer:
	"""Text summarization using Hugging Face transformers"""
	

	@st.cache_resource
	def _load_summarizer():
		"""Load summarization model"""
		try:
			summarizer = pipeline("summarization", 
								model="facebook/bart-large-cnn",
								max_length=150, 
								min_length=30,
								do_sample=False)
			return summarizer, True
		except:
			try:
				# Fallback to smaller model
				summarizer = pipeline("summarization",
									model="sshleifer/distilbart-cnn-12-6",
									max_length=120,
									min_length=25)
				return summarizer, True
			except Exception as e:
				st.warning(f"Summarization model not available\n\nReason:\n{e}")
				return None, False

	def __init__(self):
		self.summarizer, self.available = self._load_summarizer()

	@st.cache_data
	def summarize_headlines(_self, headlines):  
		"""Summarize a list of headlines"""
		if not _self.available or not headlines:
			return "Summarization not available"
	
		try:
			# Create a cache key based on headlines
			headlines_str = '|'.join(sorted(headlines))
			
			# Shuffle headlines to get random selection
			headlines_copy = headlines.copy()
			random.shuffle(headlines_copy)

			combined_text = ""
			for headline in headlines_copy:
				if len(combined_text) >= 1000:
					break
				combined_text += headline.strip() + ". "
		
			combined_text = combined_text.strip()

			if len(combined_text) < 50:
				return "Insufficient text for summarization"

			# Limit input for summarizer (most models prefer <1024 tokens ~ 1000–1500 characters)
			if len(combined_text) > 1000:
				combined_text = combined_text[:1000]

			summary = _self.summarizer(combined_text)[0]['summary_text']
			return summary

		except Exception as e:
			return f"Summarization error: {e}"

class DataVisualizer:
	"""Advanced data visualization for insights"""
	
	def __init__(self):
		plt.style.use('seaborn-v0_8')
	
	@st.cache_data
	def create_wordcloud(_self, text_data, exclude_words=None, colormap='viridis'):
		"""Generate customizable word cloud"""
		if exclude_words is None:
			exclude_words = set(['news', 'says', 'new', 'get', 'make', 'take'])
		
		# Convert set to sorted list for consistent caching
		exclude_words = sorted(list(exclude_words))
		
		# Clean and combine text
		text = ' '.join(text_data).lower()
		text = re.sub(r'[^\w\s]', ' ', text)
		
		# Remove common words
		words = [word for word in text.split() if len(word) > 1 and word not in exclude_words]
		text = ' '.join(words)
		
		if not text.strip():
			return None
			
		wordcloud = WordCloud(
			width=800, 
			height=400, 
			background_color='white',
			colormap=colormap,
			max_words=50,
			relative_scaling=0.5,
			random_state=42
		).generate(text)
		
		return wordcloud
	
	@st.cache_data
	def plot_sentiment_distribution(_self, sentiments):
		"""Create sentiment distribution bar chart"""
		sentiment_counts = Counter(sentiments)
		
		fig = go.Figure(data=[
			go.Bar(
				x=list(sentiment_counts.keys()),
				y=list(sentiment_counts.values()),
				marker_color=['#ff4444', '#ff8800', '#888888', '#44aa44', '#00aa00']
			)
		])
		
		fig.update_layout(
			title="Sentiment Distribution",
			xaxis_title="Sentiment",
			yaxis_title="Count",
			template="plotly_white"
		)
		
		return fig

class DataExporter:
	"""Handle data export in multiple formats"""
	
	@staticmethod
	def to_csv(df):
		"""Export DataFrame to CSV"""
		return df.to_csv(index=False)
	
	@staticmethod
	def to_json(df):
		"""Export DataFrame to JSON"""
		return df.to_json(orient='records', date_format='iso')
	
	@staticmethod
	def wordcloud_to_png(wordcloud):
		"""Convert wordcloud to PNG bytes"""
		if wordcloud is None:
			return None
			
		img_buffer = BytesIO()
		wordcloud.to_image().save(img_buffer, format='PNG')
		return img_buffer.getvalue()
		
	   
	  
	 
	
   
  
 













# dir\app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import re
from collections import Counter
import matplotlib.pyplot as plt
from utils import NewsDataCollector, SentimentAnalyzer, TextSummarizer, DataVisualizer, DataExporter
import hashlib

@st.cache_resource
def get_analyzers():
	"""Initialize and cache the analyzer objects"""
	collector = NewsDataCollector()
	analyzer = SentimentAnalyzer()
	summarizer = TextSummarizer()
	visualizer = DataVisualizer()
	return collector, analyzer, summarizer, visualizer

@st.cache_data
def process_sentiment_analysis(titles):
	"""Cache sentiment analysis results"""
	_, analyzer, _, _ = get_analyzers()
	results = []
	for title in titles:
		sentiment = analyzer.analyze_text(title)
		results.append(sentiment)
	return results

@st.cache_data
def generate_keyword_analysis(titles):
	"""Cache keyword analysis"""
	all_text = ' '.join(titles).lower()
	words = re.findall(r'\b\w+\b', all_text)
	word_freq = Counter([w for w in words if len(w) > 3])
	return word_freq.most_common(10)

def main():
	"""Main Streamlit application"""
	
	# Page configuration
	st.set_page_config(
		page_title="MarketPulse Pro",
		page_icon="📈",
		layout="wide",
		initial_sidebar_state="expanded"
	)
	
	# Custom CSS for better styling
	st.markdown("""
	<style>
	.main-header {
		text-align: center;
		padding: 2rem 0;
		background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
		color: white;
		margin-bottom: 2rem;
		border-radius: 10px;
	}
	.metric-card {
		background: #f0f2f6;
		padding: 1rem;
		border-radius: 10px;
		border-left: 4px solid #667eea;
	}
	</style>
	""", unsafe_allow_html=True)
	
	# Header
	st.markdown("""
	<div class="main-header">
		<h1>📈 MarketPulse Pro</h1>
		<p>Advanced News Analysis & Sentiment Intelligence Platform</p>
	</div>
	""", unsafe_allow_html=True)
	
	# Initialize components
	collector, analyzer, summarizer, visualizer = get_analyzers()
	
	# Sidebar configuration
	st.sidebar.header("🔧 Configuration")
	
	# Search parameters
	query = st.sidebar.text_input("Search Query", value="artificial intelligence", 
								 help="Enter keywords to search for")
	
	region = st.sidebar.selectbox("Region", 
								 options=['US', 'UK', 'CA', 'AU', 'NG', 'IN', 'DE', 'FR'],
								 help="Select geographical region")
	
	category_map = {
		'General': None,
		'Business': 'BUSINESS',
		'Technology': 'TECHNOLOGY', 
		'Health': 'HEALTH',
		'Science': 'SCIENCE',
		'Sports': 'SPORTS'
	}
	
	category = st.sidebar.selectbox("Category", options=list(category_map.keys()))
	
	max_articles = st.sidebar.slider("Max Articles", min_value=10, max_value=100, value=50)
	
	# Filtering options
	st.sidebar.header("🔍 Filtering")
	keyword_filter = st.sidebar.text_input("Filter by Keywords", 
										  help="Filter headlines containing these words")
	
	# Visualization options
	st.sidebar.header("🎨 Visualization")
	exclude_words = st.sidebar.text_area("Exclude Words from WordCloud", 
										value="news, says, new, get, make",
										help="Comma-separated words to exclude")
	
	colormap = st.sidebar.selectbox("WordCloud Color Scheme", 
								   options=['viridis', 'plasma', 'inferno', 'magma', 'Blues'])
	
	# Main content
	if st.sidebar.button("🚀 Analyze News", type="primary"):
		
		with st.spinner("Collecting news data..."):
			# Collect news data
			articles = collector.collect_news_data(
				query=query if query else None,
				region=region,
				category=category_map[category],
				max_articles=max_articles
			)
		
		if not articles:
			st.error("No articles found. Try adjusting your search parameters.")
			return
		
		# Convert to DataFrame
		df = pd.DataFrame(articles)
		
		# Apply keyword filtering
		if keyword_filter:
			keywords = [k.strip().lower() for k in keyword_filter.split(',')]
			mask = df['title'].str.lower().str.contains('|'.join(keywords), na=False)
			df = df[mask]
		
		if df.empty:
			st.warning("No articles match your filter criteria.")
			return
		
		# Perform sentiment analysis
		with st.spinner("Analyzing sentiment..."):
			titles = df['title'].tolist()
			sentiment_results = process_sentiment_analysis(titles)
	
			# Show visual progress
			progress_bar = st.progress(0)
			for idx in range(len(titles)):
				progress_bar.progress((idx + 1) / len(titles))
			progress_bar.empty()
		
		# Add sentiment data to DataFrame
		sentiment_df = pd.DataFrame(sentiment_results)
		df = pd.concat([df, sentiment_df], axis=1)
		
		# Display metrics
		col1, col2, col3, col4, col5 = st.columns(5)
		
		sentiment_counts = df['sentiment_label'].value_counts()
		total_articles = len(df)

		with col1:
			st.metric("Total Articles", total_articles)
		
		with col2:
			positive_pct = (sentiment_counts.get('Positive', 0) / total_articles) * 100
			st.metric("Positive %", f"{positive_pct:.1f}%")

		with col3:
			neutral_pct = (sentiment_counts.get('Neutral', 0) / total_articles) * 100
			st.metric("Neutral %", f"{neutral_pct:.1f}%")

		with col4:
			negative_pct = (sentiment_counts.get('Negative', 0) / total_articles) * 100
			st.metric("Negative %", f"{negative_pct:.1f}%")
		
		with col5:
			top_source = df['source'].value_counts().index[0] if total_articles > 0 else "N/A"
			st.metric("Top Source", top_source)
		
		# Tabs for different views
		tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Visualizations", "📝 Summary", "📋 Data", "💾 Export"])
		
		with tab1:
			st.header("Headlines Overview")
			
			# Display recent headlines with sentiment
			for _, row in df.head(10).iterrows():
				sentiment_color = {
					'Positive': 'green', 
					'Neutral': 'gray',
					'Negative': 'red'
				}.get(row['sentiment_label'], 'gray')
				
				st.markdown(f"""
				<div class="metric-card">
					<h4>{row['title']}</h4>
					<p><strong>Source:</strong> {row['source']} | 
					<strong>Sentiment:</strong> <span style="color: {sentiment_color}">{row['sentiment_label']}</span></p>
				</div>
				""", unsafe_allow_html=True)
				st.markdown("<br>", unsafe_allow_html=True)
		
		with tab2:
			st.header("Data Visualizations")
			
			col, = st.columns(1)
			
			with col:
				# Sentiment distribution
				fig_sentiment = visualizer.plot_sentiment_distribution(df['sentiment_label'])
				st.plotly_chart(fig_sentiment, use_container_width=True)
			
			# Word cloud
			st.subheader("Word Cloud")
			exclude_list = [w.strip() for w in exclude_words.split(',') if w.strip()]
			wordcloud = visualizer.create_wordcloud(df['title'].tolist(), 
												   exclude_words=set(exclude_list),
												   colormap=colormap)
			
			if wordcloud:
				fig, ax = plt.subplots(figsize=(12, 6))
				ax.imshow(wordcloud, interpolation='bilinear')
				ax.axis('off')
				st.pyplot(fig)
			else:
				st.warning("Could not generate word cloud")
		
		with tab3:
			st.header("AI-Generated Summary")
			
			if summarizer.available:
				with st.spinner("Generating summary..."):
					summary = summarizer.summarize_headlines(df['title'].tolist())
					st.info(summary)
			else:
				st.warning("Summary feature not available")
			
			# Top keywords
			st.subheader("Top Keywords")
			
			top_keywords = generate_keyword_analysis(df['title'].tolist())
			keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
			st.dataframe(keyword_df, use_container_width=True)
		
		with tab4:
			st.header("Raw Data")
			
			# Display options
			show_columns = st.multiselect(
				"Select columns to display",
				options=df.columns.tolist(),
				default=['title', 'source', 'sentiment_label']
			)
			
			if show_columns:
				st.dataframe(df[show_columns], use_container_width=True)
		
		with tab5:
			st.header("Export Data")
			
			col1, col2, col3 = st.columns(3)
			
			file_name = f"marketpulse_data_{datetime.utcnow().strftime('%Y%m%d_%H%M')}-UTC"
			with col1:
				# CSV Export
				csv_data = DataExporter.to_csv(df)
				st.download_button(
					label="📄 Download CSV",
					data=csv_data,
					file_name=f"{file_name}.csv",
					mime="text/csv"
				)
			
			with col2:
				# JSON Export
				json_data = DataExporter.to_json(df)
				st.download_button(
					label="📋 Download JSON",
					data=json_data,
					file_name=f"{file_name}.json",
					mime="application/json"
				)
			
			with col3:
				# WordCloud PNG Export
				if 'wordcloud' in locals() and wordcloud:
					png_data = DataExporter.wordcloud_to_png(wordcloud)
					if png_data:
						st.download_button(
							label="🖼️ Download WordCloud",
							data=png_data,
							file_name=f"{file_name}.png",
							mime="image/png"
						)
	
	# Footer
	st.markdown("---")
	st.markdown("""
	<div style="text-align: center; color: #666;">
		<p>🚀 MarketPulse Pro - Built for Business Intelligence</p>
		<p>Powered by Advanced NLP & Real-time Data Analysis</p>
	</div>
	""", unsafe_allow_html=True)

if __name__ == "__main__":
	main()
