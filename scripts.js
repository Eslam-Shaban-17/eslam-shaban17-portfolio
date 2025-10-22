// Projects Data - Each project can have multiple categories
const projects = [
    // Comprehensive End-to-End ML Project
    {
        title: "Fleet Management KPI Analytics",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Interactive Power BI dashboard for fleet management operations. Tracks key performance indicators including vehicle utilization, maintenance schedules, fuel consumption, driver performance, and operational costs with real-time monitoring.",
        tech: ["Power BI", "DAX", "KPI Tracking", "Dashboard Design"],
        image: "./imgs/fleet-pbi.png",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/02%20-%20Power%20BI%20Projects/03%20-%20Fleet%20Management%20kPI%20Analytics"
    },
    {
        title: "Customer Churn Prediction",
        categories: ["machine-learning", "business-intelligence"],
        primaryCategory: "Machine Learning & AI",
        description: "Comprehensive ML project to predict customer churn in telecommunications using multiple algorithms (Logistic Regression, Random Forest, XGBoost). Achieved 80%+ accuracy with business insights and actionable recommendations.",
        tech: ["Python", "Scikit-learn", "XGBoost", "Pandas", "SMOTE", "Feature Engineering"],
        image: "./imgs/customer-churn.jpg",
        link: "https://github.com/Eslam-Shaban-17/Customer-Churn-Prediction"
    },
    {
        title: "HR Analytics Dashboard",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Comprehensive HR analytics Power BI dashboard with three key modules: Demographics analysis (headcount, diversity, age distribution), Performance metrics (employee ratings, productivity trends), and Executive Overview (turnover rate, hiring trends, department insights) for strategic workforce planning.",
        tech: ["Power BI", "DAX", "HR Analytics", "KPI Tracking", "Workforce Planning"],
        image: "./imgs/hr-pbi.png",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/02%20-%20Power%20BI%20Projects/04%20-%20HR%20Full%20Project%20with%20Power%20BI"
    },
    {
        title: "AdventureWorks End-to-End Analytics",
        categories: ["business-intelligence", "sql", "data-analytics", "data-engineering"],
        primaryCategory: "Business Intelligence",
        description: "Comprehensive BI solution analyzing sales, customers, supply chain, HR, and financial performance. Started by solving business requirements using advanced SQL (CTEs, Window Functions: NTILE, LEAD, LAG, RANK), then designed dimensional star schema data model, and built 100+ DAX measures including time intelligence for 10 interactive Power BI dashboards. Features RFM customer segmentation on 120K+ transactions.",
        tech: ["Power BI", "SQL Server", "Advanced SQL", "DAX", "Star Schema", "Window Functions", "Time Intelligence"],
        image: "./imgs/BI.png",
        link: "https://github.com/Eslam-Shaban-17/ADW-Mega-Project"
    },
    {
        title: "Customer & Sales Insights Dashboard",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Comprehensive Excel dashboard analyzing customer behavior, sales performance, and operational efficiency. Features COGS calculations, top customer analysis, regional performance metrics, shipment mode optimization, and delivery time tracking to enhance customer loyalty and business performance.",
        tech: ["Excel", "Pivot Tables", "Data Analysis", "Business Metrics", "Dashboard Design"],
        image: "./imgs/excell-dashboard-revenue.png",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/04%20-%20Excel/02%20-%20Customer%20%26%20Sales%20Insights%20Dashboard"
    },
    {
        title: "Awesome Chocolates Sales Analysis",
        categories: ["business-intelligence", "data-analytics", "sql"],
        primaryCategory: "Business Intelligence",
        description: "Power BI + MySQL integrated analytics project analyzing Awesome Chocolates sales performance. Features relational data modeling across 5 tables, interactive dashboards showing regional performance, top-selling products (Milk Bars, 85% Dark Bars), team analytics, and revenue trends with December peak insights.",
        tech: ["Power BI", "MySQL", "SQL", "Data Modeling", "DAX", "Relational Databases"],
        image: "./imgs/chocolate-pbi.png",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/02%20-%20Power%20BI%20Projects/05%20-%20Chocolate%20Sales%20analytics"
    },
    {
        title: "Monthly Sales KPIs Dashboard",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Comprehensive Power BI dashboard tracking monthly sales KPIs with geographic logistics visualization. Features interactive maps for distribution analysis, hierarchical category tree visualization for product classification, and performance metrics across goods categories for supply chain optimization.",
        tech: ["Power BI", "DAX", "Geographic Mapping", "Tree Visualization", "KPI Tracking"],
        image: "./imgs/monthly-kpi-pbi.png",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/02%20-%20Power%20BI%20Projects/06%20-%20Monthly%20Sales%20KPI's"
    },
    {
        title: "Superstore Logistics & Supply Chain",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Advanced Power BI dashboard for superstore logistics and supply chain management. Features comprehensive KPI tracking, geographic mapping for logistics routes, distribution center analysis, hierarchical category tree visualization for inventory management, and supply chain performance metrics for operational optimization.",
        tech: ["Power BI", "DAX", "Supply Chain Analytics", "Logistics Mapping", "Inventory Management"],
        image: "./imgs/logistics-pbi.png",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/02%20-%20Power%20BI%20Projects/07%20-%20Superstore%20Logistic"
    },
    {
        title: "E-Commerce Data Warehouse & ETL Pipeline",
        categories: ["data-engineering", "sql", "business-intelligence"],
        primaryCategory: "Data Engineering & ETL",
        description: "Enterprise-grade data warehouse solution processing 112,650+ orders from Brazilian E-Commerce dataset. Features automated SSIS ETL pipelines with SCD Type 2, star schema design, incremental loading (80% faster), data quality framework (99.9% accuracy), and reduced reporting time from days to seconds. Includes comprehensive monitoring and error handling.",
        tech: ["SQL Server", "SSIS", "ETL", "Star Schema", "SCD Type 2", "Data Warehouse", "Data Quality"],
        image: "./imgs/ETL-Process.png",
        link: "https://github.com/Eslam-Shaban-17/E-commerce-Sales-Data-Warehouse-ETL"
    },
    // End-to-End Projects with ML/DL
    {
        title: "Bob Chatbot - Egyptian Food Ordering",
        categories: ["machine-learning", "nlp", "data-engineering"],
        primaryCategory: "Natural Language Processing",
        description: "End-to-end chatbot development NLP project for an online food delivery system for Egyptian food using FastAPI backend, MySQL database, and DialogFlow.",
        tech: ["Python", "FastAPI", "NLP", "DialogFlow", "MySQL"],
        image: "./imgs/chatbot.png",
        link: "https://github.com/Eslam-shaban/bob-chatbot-egyptian-food"
    },
    {
        title: "YouTube Analytics Data Engineering",
        categories: ["data-engineering", "data-analytics"],
        primaryCategory: "Data Engineering & ETL",
        description: "End-to-end data engineering project to securely manage, streamline, and analyze YouTube videos data based on video categories and trending metrics using AWS services.",
        tech: ["AWS S3", "AWS Glue", "AWS Lambda", "AWS Athena", "QuickSight", "Python"],
        image: "./imgs/youtube_analytics_dashboard.jpeg",
        link: "https://github.com/Eslam-shaban/data-engineering-youtube-analytics"
    },
    {
        title: "Football Players Image Classification",
        categories: ["deep-learning", "machine-learning"],
        primaryCategory: "Deep Learning & CV",
        description: "End-to-end image classification using OpenCV face detection and machine learning. Built a complete website with Flask, HTML, CSS, and JavaScript.",
        tech: ["Python", "OpenCV", "Machine Learning", "Flask", "HTML/CSS/JS"],
        image: "./imgs/sports_img_2.jpg",
        link: "https://github.com/Eslam-shaban/football_players_image_classification"
    },
    {
        title: "Bangalore House Price Prediction",
        categories: ["machine-learning"],
        primaryCategory: "Machine Learning & AI",
        description: "End-to-end real estate price prediction based on bedrooms, kitchen, and bathroom. Built using machine learning and deployed as a web application.",
        tech: ["Python", "Machine Learning", "Flask", "HTML/CSS/JS"],
        image: "./imgs/bangolare_2_.png",
        link: "https://github.com/Eslam-shaban/Banglore_home_prices_Render"
    },
    {
        title: "Social Network Analysis Desktop App",
        categories: ["data-analytics", "machine-learning"],
        primaryCategory: "Data Analytics",
        description: "Desktop software for network analysis and visualization. Enables users to interact with graphs, color and adjust nodes/edges based on calculated metrics.",
        tech: ["Python", "Tkinter", "NetworkX", "Graph Analysis"],
        image: "./imgs/socialnet.png",
        link: "https://github.com/Eslam-shaban/SocialMediaAnalysis_TkinterPython"
    },

    // Deep Learning & Computer Vision Projects
    {
        title: "Facial Emotion Recognition",
        categories: ["deep-learning"],
        primaryCategory: "Deep Learning & CV",
        description: "Multi-class classification for facial emotion recognition using Convolutional Neural Network (CNN) to detect different emotions.",
        tech: ["Python", "CNN", "TensorFlow", "Keras", "Computer Vision"],
        image: "./imgs/Emotion-Detection-1.png",
        link: "https://www.kaggle.com/code/eslamshaban/facial-emotion-recognition-multi-classification"
    },
    {
        title: "Gender and Age Detection (CNN)",
        categories: ["deep-learning"],
        primaryCategory: "Deep Learning & CV",
        description: "Gender and age detection using Convolutional Neural Network for accurate demographic prediction from images.",
        tech: ["Python", "CNN", "Deep Learning", "TensorFlow"],
        image: "./imgs/gender-age-detection.jpg",
        link: "https://www.kaggle.com/code/eslamshaban/gender-age-detection"
    },
    {
        title: "Gender & Age Detection (OpenCV)",
        categories: ["deep-learning"],
        primaryCategory: "Deep Learning & CV",
        description: "Real-time gender and age detection from camera, video, or photo using OpenCV face detection techniques.",
        tech: ["Python", "OpenCV", "Computer Vision"],
        image: "./imgs/gender-age-detection-opencv.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/tree/main/Data%20Science%20Projects/Kaggle%20Projects/06%20-%20Gender%20and%20Age%20Detection%20OpenCV"
    },
    {
        title: "Dogs vs Cats Classification",
        categories: ["deep-learning"],
        primaryCategory: "Deep Learning & CV",
        description: "Binary image classification using Convolutional Neural Network to distinguish between dogs and cats with high accuracy.",
        tech: ["Python", "CNN", "TensorFlow", "Keras"],
        image: "./imgs/catvsdog1.jpg",
        link: "https://www.kaggle.com/code/eslamshaban/dogs-vs-cats-classification"
    },
    {
        title: "Intel Image Scene Classification",
        categories: ["deep-learning"],
        primaryCategory: "Deep Learning & CV",
        description: "Multi-class image scene classification using CNN and Transfer Learning with MobileNet for improved accuracy.",
        tech: ["Python", "CNN", "Transfer Learning", "MobileNet"],
        image: "./imgs/intel_img.png",
        link: "https://www.kaggle.com/code/eslamshaban/image-scene-classification-of-multiclass"
    },
    {
        title: "MNIST Digit Recognizer",
        categories: ["deep-learning"],
        primaryCategory: "Deep Learning & CV",
        description: "Handwritten digit recognition using Convolutional Neural Network with data augmentation for improved performance.",
        tech: ["Python", "CNN", "TensorFlow", "Data Augmentation"],
        image: "./imgs/mnist.png",
        link: "https://www.kaggle.com/code/eslamshaban/mnist-cnn-datagen"
    },
    {
        title: "Flowers Classification with VGG16",
        categories: ["deep-learning"],
        primaryCategory: "Deep Learning & CV",
        description: "Multi-class flower classification (chamomile, tulip, rose, sunflower, dandelion) using VGG16 Transfer Learning.",
        tech: ["Python", "Transfer Learning", "VGG16", "Deep Learning"],
        image: "./imgs/flowers_regression.png",
        link: "https://www.kaggle.com/code/eslamshaban/ds-flower-regression-vgg16"
    },

    // Machine Learning Projects
    {
        title: "House Prices Prediction",
        categories: ["machine-learning"],
        primaryCategory: "Machine Learning & AI",
        description: "Advanced regression techniques for predicting house sales prices. Practice feature engineering, Random Forests, and gradient boosting.",
        tech: ["Python", "Regression", "Feature Engineering", "XGBoost"],
        image: "./imgs/house_price_prediction.jpg",
        link: "https://www.kaggle.com/code/eslamshaban/housepriceprediction"
    },
    {
        title: "Car Price Prediction",
        categories: ["machine-learning"],
        primaryCategory: "Machine Learning & AI",
        description: "Car price prediction using multiple regression techniques including Linear Regression, Decision Tree, Random Forest, and SGD Regressor.",
        tech: ["Python", "Regression", "Scikit-learn", "Machine Learning"],
        image: "./imgs/car_price.jpg",
        link: "https://www.kaggle.com/code/eslamshaban/car-price-regression-techniques"
    },
    {
        title: "Titanic Survival Prediction",
        categories: ["machine-learning"],
        primaryCategory: "Machine Learning & AI",
        description: "Classic machine learning classification problem to predict survival on the Titanic using passenger data.",
        tech: ["Python", "Classification", "Machine Learning", "Pandas"],
        image: "./imgs/titanic.jpg",
        link: "https://www.kaggle.com/eslamshaban/classificationtitanicdataset"
    },
    {
        title: "Credit Card Approval Prediction",
        categories: ["machine-learning", "data-analytics"],
        primaryCategory: "Machine Learning & AI",
        description: "Machine learning model to predict credit card application approval with data cleaning, preprocessing, and model evaluation.",
        tech: ["Python", "Machine Learning", "Classification", "Scikit-learn"],
        image: "./imgs/creditcard.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Science%20Projects/Data%20Camp%20Projects/06%20-%20Predicting%20Credit%20Card%20Approvals/Predicting_credit_approvals.ipynb"
    },
    {
        title: "Iris Flowers Clustering",
        categories: ["machine-learning", "data-analytics"],
        primaryCategory: "Machine Learning & AI",
        description: "Unsupervised machine learning using K-Means clustering on the Iris dataset with data visualization.",
        tech: ["Python", "K-Means", "Unsupervised ML", "Clustering"],
        image: "./imgs/iris img1.png",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Spark_Internship/Task2%20Unsupervised%20ML.ipynb"
    },

    // NLP & Text Analysis Projects
    {
        title: "Movie Recommendation System (TF-IDF)",
        categories: ["nlp", "machine-learning"],
        primaryCategory: "Natural Language Processing",
        description: "Content-based recommendation system using TF-IDF vectorization on TMDB 5000 movie dataset.",
        tech: ["Python", "NLP", "TF-IDF", "Recommendation System"],
        image: "./imgs/recommender_tmdb 5000.png",
        link: "https://www.kaggle.com/code/eslamshaban/content-based-recommendation-using-tf-idf"
    },
    {
        title: "Sentiment Analysis (NLP)",
        categories: ["nlp", "machine-learning"],
        primaryCategory: "Natural Language Processing",
        description: "Natural Language Processing case study for sentiment analysis on restaurant reviews to extract customer opinions.",
        tech: ["Python", "NLP", "Sentiment Analysis", "NLTK"],
        image: "./imgs/Sentiment-Analysis-Examples-Social.png",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Science%20Projects/Kaggle%20Projects/05%20-%20Sentiment%20Analysis_NLP/Sentiment%20Analysis%20case%20study%20(Nlp).ipynb"
    },

    // Data Analysis Projects
    {
        title: "McDonald's Sales & Profit Analysis",
        categories: ["data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Excel-based analytical report examining McDonald's sales and profit performance across different time periods and geographic regions. Features temporal trend analysis, regional performance comparison, profitability metrics, and visual insights for strategic decision-making.",
        tech: ["Excel", "Pivot Tables", "Data Analysis", "Time Series Analysis", "Regional Analytics"],
        image: "./imgs/mc-donalds-excel.png",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/04%20-%20Excel/04%20-%20McDonald's%20Report"
    },
    {
        title: "Coffee Sales Dashboard",
        categories: [ "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Interactive Excel dashboard analyzing coffee sales with multi-dimensional filtering. Features roast type analysis (Dark, Medium, Light), coffee variety breakdown (Arabica, Robusta, Liberica, Excelsa), regional performance metrics, and customer segmentation for comprehensive business insights.",
        tech: ["Excel", "Pivot Tables", "Interactive Filters", "Data Visualization", "Slicers"],
        image: "./imgs/excel.jpg",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/04%20-%20Excel/05%20-%20Coffe%20Sales%20Dashboard"
    },
    {
        title: "Movie Industry Dataset Correlation",
        categories: ["data-analytics"],
        primaryCategory: "Data Analytics",
        description: "Data correlation analysis on movie industry dataset with data cleaning, manipulation, and visualization to understand industry trends.",
        tech: ["Python", "Pandas", "Data Visualization", "Correlation Analysis"],
        image: "./imgs/movies.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Analysis%20Projects/01%20-%20Python%20Projects/02%20-%20Correlation%20in%20Python%20-%20Movie%20Industry%20Dataset/Data%20Correlation%20on%20Movie%20Data.ipynb"
    },
    {
        title: "San Francisco Salaries Analysis",
        categories: ["data-analytics"],
        primaryCategory: "Data Analytics",
        description: "Exploratory data analysis on San Francisco employee salaries with data cleaning and visualization.",
        tech: ["Python", "Pandas", "Data Analysis", "Matplotlib"],
        image: "./imgs/san-francisco-salaries.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Analysis%20Projects/01%20-%20Python%20Projects/01%20-%20Data%20Exploration%20SF%20Salaries%20with%20Python/data-exploration-sf-salaries-v1.ipynb"
    },
    {
        title: "Amazon Web Scraping",
        categories: ["web-scraping", "data-analytics"],
        primaryCategory: "Web Scraping & APIs",
        description: "Web scraping project to extract Data Analysis book titles, prices, and ratings from Amazon using Python.",
        tech: ["Python", "Web Scraping", "BeautifulSoup", "Requests"],
        image: "./imgs/amazon2.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Analysis%20Projects/01%20-%20Python%20Projects/03%20-%20Web%20Scraping%20Python%20-%20Amazon/Amazon%20Web%20Scraping.ipynb"
    },
    {
        title: "Time Series Analysis Case Study",
        categories: ["data-analytics", "machine-learning"],
        primaryCategory: "Data Analytics",
        description: "Time series analysis to understand seasonality and trends in data using Python statistical techniques.",
        tech: ["Python", "Time Series", "Statistical Analysis", "Pandas"],
        image: "./imgs/Time-Series-Analysis.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Analysis%20Projects/01%20-%20Python%20Projects/05%20-%20Python%20Time-series%20Analysis%20case%20study/Time%20Series%20Case%20Study.ipynb"
    },
    {
        title: "Wuzzuf Job Market Web Scraping",
        categories: ["web-scraping", "data-analytics"],
        primaryCategory: "Web Scraping & APIs",
        description: "Web scraping Wuzzuf.com to extract Data Scientist job titles and company information for market analysis.",
        tech: ["Python", "Web Scraping", "BeautifulSoup", "Data Analysis"],
        image: "./imgs/wuzzuf-425x215.png",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Analysis%20Projects/01%20-%20Python%20Projects/06%20-%20Web%20scraping%20Wuzzuf%20website/web%20scraping%20with%20python%20for%20wuzzuf.com.ipynb"
    },
    {
        title: "Crypto Website API Automation",
        categories: ["web-scraping", "data-analytics"],
        primaryCategory: "Web Scraping & APIs",
        description: "Automating cryptocurrency website API data pull using Python for real-time market data collection.",
        tech: ["Python", "API", "Automation", "Cryptocurrency"],
        image: "./imgs/coinmarket.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Analysis%20Projects/01%20-%20Python%20Projects/04%20-%20Automating%20Crypto%20Website%20API%20Pull%20Using%20Python/website%20API%20pull.ipynb"
    },
    {
        title: "Netflix Movies Analysis",
        categories: ["data-analytics"],
        primaryCategory: "Data Analytics",
        description: "Data manipulation and visualization of Netflix movie and TV show data using foundational Python skills.",
        tech: ["Python", "Pandas", "Data Visualization", "Matplotlib"],
        image: "./imgs/Netflix.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Science%20Projects/Data%20Camp%20Projects/01%20-%20Investigating%20Netflix%20Movies%20and%20Guest%20Stars%20in%20The%20Office/Netflix_notebook.ipynb"
    },
    {
        title: "Nobel Prize Winners Analysis",
        categories: ["data-analytics"],
        primaryCategory: "Data Analytics",
        description: "Explore a century of Nobel Prize winners data with data cleaning, manipulation, and visualization.",
        tech: ["Python", "Pandas", "Data Analysis", "Visualization"],
        image: "./imgs/Nobel.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Science%20Projects/Data%20Camp%20Projects/04%20-%20A%20Visual%20History%20of%20Nobel%20Prize%20Winners/Nobel_Price.ipynb"
    },
    {
        title: "Google Play Store Analysis",
        categories: ["data-analytics"],
        primaryCategory: "Data Analytics",
        description: "Load, clean, and visualize scraped Google Play Store data to gain insights into the Android app market.",
        tech: ["Python", "Pandas", "Data Cleaning", "Visualization"],
        image: "./imgs/playstore.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Science%20Projects/Data%20Camp%20Projects/02%20-%20The%20Android%20App%20Market%20on%20Google%20Play/Android_Apps.ipynb"
    },
    {
        title: "GitHub History of Scala",
        categories: ["data-analytics"],
        primaryCategory: "Data Analytics",
        description: "Find true Scala experts by exploring development history in Git and GitHub with data analysis.",
        tech: ["Python", "Git Analysis", "Data Manipulation", "Pandas"],
        image: "./imgs/scala_github.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Science%20Projects/Data%20Camp%20Projects/03%20-%20The%20GitHub%20History%20of%20the%20Scala%20Language/GitHub_history_of_scala.ipynb"
    },
    {
        title: "Marketing Case Study",
        categories: ["data-analytics"],
        primaryCategory: "Data Analytics",
        description: "Marketing analytics examining user time on app/website, length of membership, and spending patterns.",
        tech: ["Python", "Marketing Analytics", "Statistical Analysis"],
        image: "./imgs/startup-marketing.png",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Science%20Projects/Kaggle%20Projects/02%20-%20Marketing%20Case%20study/Marketing%20Case%20study.ipynb"
    },

    // SQL Projects
    {
        title: "COVID-19 Data Exploration (SQL)",
        categories: ["sql", "data-analytics"],
        primaryCategory: "SQL & Databases",
        description: "Comprehensive SQL data exploration and analysis of coronavirus deaths data with advanced queries.",
        tech: ["SQL", "Data Analysis", "Data Exploration"],
        image: "./imgs/covid.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Analysis%20Projects/02%20-%20SQL%20Projects/01%20-%20Data%20Exploration%20in%20SQL%20-%20Coronavirus%20(COVID-19)%20Deaths/SQL_data_exploration_Project_covid.sql"
    },
    {
        title: "Nashville Housing Data Cleaning (SQL)",
        categories: ["sql", "data-analytics"],
        primaryCategory: "SQL & Databases",
        description: "Advanced SQL data cleaning techniques applied to Nashville housing dataset.",
        tech: ["SQL", "Data Cleaning", "Data Transformation"],
        image: "./imgs/nashville.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Analysis%20Projects/02%20-%20SQL%20Projects/02%20-%20Data%20Cleaning%20in%20SQL%20-%20Nashville%20Housing%20Data/Data%20Cleaning%20Portfolio%20Project%20Queries.sql"
    },

    // Business Intelligence & Dashboard Projects (Tableau, Power BI)
    {
        title: "Airbnb Dashboard (Tableau)",
        categories: ["business-intelligence"],
        primaryCategory: "Business Intelligence",
        description: "Interactive Tableau dashboard analyzing Airbnb open data with sales analysis and data visualization.",
        tech: ["Tableau", "Data Visualization", "Dashboard Design"],
        image: "./imgs/airbnb.jpg",
        link: "https://public.tableau.com/app/profile/eslam170/viz/AirbnbProject_16490884258440/AirbnbDashboard"
    },
    {
        title: "Job Market Analysis (Power BI)",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "End-to-end case study analyzing job market data with data cleaning, manipulation, and interactive Power BI dashboard.",
        tech: ["Power BI", "DAX", "Data Analysis", "Dashboard"],
        image: "./imgs/job-market.jpg",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Visualization%20Projects/02%20-%20Power%20BI%20Projects/01%20-%20Data%20Visualization%20with%20PowerBI%20-%20Analysing%20Job%20Market%20Data/Case%20study%20analysing%20job%20market%20data.pdf"
    },
    {
        title: "COVID Deaths Dashboard (Tableau)",
        categories: ["business-intelligence", "data-analytics", "sql"],
        primaryCategory: "Business Intelligence",
        description: "End-to-end COVID-19 analysis project. Explored deaths and vaccinations datasets using SQL Server with advanced techniques (Joins, CTEs, Temp Tables, Window Functions). Calculated key metrics including death rate, infection percentage, and vaccination progress. Created reusable views and aggregate queries, then built comprehensive Tableau dashboard with interactive filters for visualization.",
        tech: ["Tableau", "SQL Server", "CTEs", "Window Functions", "Temp Tables", "Data Visualization"],
        image: "./imgs/covid_visualize.jpg",
        link: "https://public.tableau.com/app/profile/eslam170/viz/CovidDeathsDashboard_16567910075040/CovidDeathsDashboard"
    },
    {
        title: "Startup Expansion Dashboard (Power BI)",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Full data analysis project combining Python data processing with Power BI visualization for startup expansion strategy.",
        tech: ["Power BI", "Python", "Business Intelligence"],
        image: "./imgs/startup-marketing.png",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Visualization%20Projects/02%20-%20Power%20BI%20Projects/02%20-%20Full%20Data%20Analysis%20Project%20using%20Python%20%26%20Power%20BI%20-%20Starups%20expansion/data/startups-expansion.pdf"
    },
    {
        title: "Company Sales Report (Tableau)",
        categories: ["business-intelligence"],
        primaryCategory: "Business Intelligence",
        description: "Real company sales dashboard with data cleaning, manipulation, and comprehensive sales visualization in Tableau.",
        tech: ["Tableau", "Sales Analytics", "Dashboard"],
        image: "./imgs/daily-weekly-sales-reports-examples.png",
        link: "https://public.tableau.com/app/profile/eslam170/viz/CompanySalesReport/sales_report"
    },
    {
        title: "Sales Analytics Dashboard (Power BI)",
        categories: ["business-intelligence"],
        primaryCategory: "Business Intelligence",
        description: "Analysis and insights about company sales, profit margins, and regional performance using Power BI.",
        tech: ["Power BI", "Sales Analytics", "DAX"],
        image: "./imgs/Sales-Report-Format-PPT-Templates-1001x436.png",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Visualization%20Projects/02%20-%20Power%20BI%20Projects/02%20-%20Data%20Visualization%20with%20Power%20BI%20-%20Company_data/Sales%20Reporting.pdf"
    },
    {
        title: "Company Sales (Google Data Studio)",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Exploratory data analysis and visualization of company sales and revenues using Google Data Studio.",
        tech: ["Google Data Studio", "Data Visualization"],
        image: "./imgs/google-data-studio-cab.jpg.png",
        link: "https://github.com/Eslam-shaban/Data-Science-and-Analysis-Projects/blob/main/Data%20Visualization%20Projects/03%20-%20Google%20Data%20studio/01%20-%20%20Data%20Visualization%20with%20Google%20Data%20studio%20-%20Company_data/Sales_Report.pdf"
    },
    {
        title: "Bike Sales Dashboard (Excel)",
        categories: ["business-intelligence", "data-analytics"],
        primaryCategory: "Business Intelligence",
        description: "Complete Excel project with data importing, cleaning, manipulation, and dashboard creation for bike sales analysis.",
        tech: ["Excel", "Data Analysis", "Pivot Tables", "Charts"],
        image: "./imgs/bikes-excel.png",
        link: "https://github.com/Eslam-Shaban-17/Data-Science-and-Analysis-Projects/tree/main/Data%20Visualization%20Projects/04%20-%20Excel/01%20-%20Bikes%20Analytics%20Report%20in%20Excel"
    }
];

// Count projects in each category
function countProjectsByCategory() {
    const counts = {
        all: projects.length,
        'machine-learning': 0,
        'deep-learning': 0,
        'nlp': 0,
        'data-engineering': 0,
        'business-intelligence': 0,
        'data-analytics': 0,
        'sql': 0,
        'web-scraping': 0
    };

    projects.forEach(project => {
        project.categories.forEach(category => {
            if (counts.hasOwnProperty(category)) {
                counts[category]++;
            }
        });
    });

    return counts;
}

// Update category counts in the UI
function updateCategoryCounts() {
    const counts = countProjectsByCategory();
    
    Object.keys(counts).forEach(category => {
        const countElement = document.getElementById(`count-${category}`);
        if (countElement) {
            countElement.textContent = counts[category];
        }
    });
}

// Render projects based on filter
function renderProjects(filter = 'all') {
    const projectGrid = document.getElementById('projectGrid');
    const projectsSection = document.getElementById('projects');
    
    // Add loading state
    projectGrid.style.opacity = '0.7';
    projectGrid.style.transition = 'opacity 0.3s ease';
    
    // Clear content
    projectGrid.innerHTML = '';

    const filteredProjects = filter === 'all' 
        ? projects 
        : projects.filter(p => p.categories.includes(filter));

    // If no projects found, show message
    if (filteredProjects.length === 0) {
        projectGrid.innerHTML = `
            <div class="no-projects-message">
                <h3>No projects found for this filter</h3>
                <p>Try selecting a different category or "All Projects"</p>
            </div>
        `;
        projectGrid.style.opacity = '1';
        return;
    }

    filteredProjects.forEach(project => {
        const card = document.createElement('div');
        card.className = 'project-card';
        card.setAttribute('data-categories', project.categories.join(' '));
        
        card.innerHTML = `
            <img src="${project.image}" alt="${project.title}" class="project-image">
            <div class="project-content">
                <span class="project-category">${project.primaryCategory}</span>
                <h3 class="project-title">${project.title}</h3>
                <p class="project-description">${project.description}</p>
                <div class="project-tech">
                    ${project.tech.map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
                </div>
                <a href="${project.link}" class="project-link" target="_blank" rel="noopener noreferrer">View Project →</a>
            </div>
        `;
        
        projectGrid.appendChild(card);
    });
    
    // Restore opacity and scroll to projects section
    setTimeout(() => {
        projectGrid.style.opacity = '1';
        // Smooth scroll to projects section to maintain context
        projectsSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }, 100);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    renderProjects();
    updateCategoryCounts();
});

// Theme Toggle
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

// Check for saved theme preference or default to dark mode
const currentTheme = localStorage.getItem('theme') || 'dark';
if (currentTheme === 'dark') {
    body.classList.add('dark-theme');
    themeToggle.textContent = '☀️';
}

themeToggle.addEventListener('click', () => {
    body.classList.toggle('dark-theme');

    if (body.classList.contains('dark-theme')) {
        themeToggle.textContent = '☀️';
        localStorage.setItem('theme', 'dark');
    } else {
        themeToggle.textContent = '🌙';
        localStorage.setItem('theme', 'light');
    }
});

// Filter functionality
const filterBtns = document.querySelectorAll('.filter-item');

filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all buttons
        filterBtns.forEach(b => b.classList.remove('active'));
        // Add active class to clicked button
        btn.classList.add('active');

        const filterValue = btn.getAttribute('data-filter');
        
        // Show loading state
        const projectGrid = document.getElementById('projectGrid');
        projectGrid.style.opacity = '0.7';
        
        // Small delay to show loading state
        setTimeout(() => {
            renderProjects(filterValue);
        }, 150);
    });
});

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
