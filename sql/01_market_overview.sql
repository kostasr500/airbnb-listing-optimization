SELECT 
    neighbourhood_cleansed AS neighbourhood,
    COUNT(*) AS total_listings,
    ROUND(AVG(price), 2) AS avg_price,
    ROUND(MEDIAN(price), 2) AS median_price
FROM 'data/processed/athens_listings_clean.csv'
GROUP BY neighbourhood_cleansed
HAVING COUNT(*) >= 30
ORDER BY avg_price DESC
LIMIT 5;
