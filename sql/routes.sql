SELECT

    --Time Series column
    DATE_FORMAT(datedriven, "%Y-%m-%d") as date,

    --Distance of all routes traveled
    SUM(distance / 1000) AS distance_km,

    --Number of utes driving that day
    COUNT(DISTINCT(rego)) AS utes_driven,

    --Day of week
    DAYOFWEEK(datedriven) AS day_of_week,

    --Weekend or Weekday
    CASE WHEN DAYOFWEEK(datedriven) IN (1, 7) THEN 1 ELSE 0 end AS is_weekend,

    --Month Driven
    MONTH(datedriven) AS month,

    --Sine wave representation of date through the seasonal year (Jan 14 summer peak)
    ROUND(0.5 - 0.5 * SIN(2 * Pi() * (DAYOFYEAR(datedriven) - 14) / 365),4) AS seasonal_scalar,

    --How long has business been running in days
    DATEDIFF(datedriven, '2023-11-11') AS days_in_business

FROM
  routes

WHERE
  --Only customer routes driven
  paid = 1
GROUP BY date
ORDER BY date asc