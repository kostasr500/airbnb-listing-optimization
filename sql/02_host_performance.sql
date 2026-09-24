SELECT 
    room_type,
    CASE 
        WHEN is_superhost = 1 THEN 'SUPERHOST'
        ELSE 'Standard'
    END AS host_status,
    COUNT(*) AS total_listings,
    ROUND(AVG(price), 2) AS avg_price,
    ROUND(MEDIAN(price), 2) AS median_price,
    ROUND(AVG(review_scores_rating), 2) AS avg_rating
FROM 'data/processed/athens_listings_clean.csv'
GROUP BY room_type, is_superhost
ORDER BY room_type, host_status DESC;
