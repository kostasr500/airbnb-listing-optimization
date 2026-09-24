WITH ranked_listings AS (
    SELECT 
        id,
        neighbourhood_cleansed AS neighbourhood,
        accommodates,
        price,
        DENSE_RANK() OVER (
            PARTITION BY neighbourhood_cleansed 
            ORDER BY price DESC
        ) AS price_rank
    FROM 'data/processed/athens_listings_clean.csv'
    WHERE room_type = 'Entire home/apt'
)
SELECT 
    neighbourhood,
    id,
    accommodates,
    price,
    price_rank
FROM ranked_listings
WHERE price_rank <= 3
ORDER BY neighbourhood, price_rank;