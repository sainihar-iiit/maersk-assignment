# [Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/)
## 🚀 Key Features
🗣️ 1. Voice & Text Query System
📈 2. KPIs (Key Performance Indicators)
⚡ 3. Quick Insights
🧠 4. AI-Driven Query Interpretation
🔄 5. Yearly, Quarterly & Group Comparisons
🔮 6. Sales Forecasting
## ⚡ Quick Insights
![Quick insights]
Quick insights will basically helps user understands various aspects of data like what columns are there in dataset
,In each category which one are performing well.Here in the given quick insights we can get quick insights on product categories,products,city,states.
(maersk_images/quickinsights.PNG)
One-click insights to instantly generate charts for:

 1.Top / Bottom 5 Product Categories
 ![top 5 product categories](maersk_images/top5productcategories.PNG)
 2.Top / Bottom 5 States
 ![states](maersk_images/top5state.PNG)
 3.Top / Bottom 5 Cities
 ![city](maersk_images/top5city.PNG)
 4.Payment Method Distribution (Pie Chart)
 ![payment](maersk_images/payment.PNG)
 Each insight opens a dynamic bar/pie chart.

## 🧠  AI-Driven Query Interpretation
In this part we can ask agent various aspects of data and agent will answer accordingly and it not only gives answers but also it will give various graphs like bargraph ,pie chart ,lines etc to represent outcome visually in diagrams so it will help user to have better experience
![agent](maersk_images/agent.PNG)
The system understands natural language and automatically:

1.Detects category (cities, states, products, sellers)
2.Detects aggregation (sum, avg, count)
3.Detects top/bottom N
4.Detects rank (3rd highest, 2nd lowest)
5.Detects time filters (months like April 2018)
6.Detects comparison queries (e.g., compare 2017 vs 2018)
7.Detects forecasting queries
#Analysis
## Compare sales in 2017 vs 2018
![comparision](maersk_images/salescomparision.PNG)
Blue line indicates year 2017  and orange line indicates year 2018 where we can see there is a sharp dip in year 2018 after august month because the data is till august 2018
## Compare sales in 2017 vs 2018 for top 10 States
![comparision_state_wise](maersk_images/statecomparision.PNG)
Here it gives comparies of top 10 states in sales in the year 2017 and 2018 where blue color indicates year 2017 and orange color indicates year 2018
## Forecasting sales for next 6 months for top 5 states in sales through voice
![forecast](maersk_images/forecast.PNG)
Here it uses prophet and tqdm libraries uses and by analysing previous trends and previous data ,it will predict or forecast next 6 months sales for top 5 states
It is giving through my voice command 
##Compare sales in Q1 2017 vs Q1 2018
Here it is comparing sales in first three months in 2017 and first three months in 2018 i.e January,February,March

