CREATE TEMP FUNCTION
  maxheartrate(x FLOAT64)
  RETURNS FLOAT64 AS( 211-0.64*x );
CREATE TEMP FUNCTION
  timeofday(x INT64)
  RETURNS STRING AS(
    CASE
      WHEN x >= 4 AND x < 12 THEN "MORNING"
      WHEN x >= 12
    AND x < 18 THEN "AFTERNOON"
      WHEN (x < 4 AND x > 0) OR x >= 18 THEN "EVENING"
    ELSE
    "N/A"
  END
    );
CREATE TEMP FUNCTION
  weekend(x DATE)
  RETURNS BOOL AS (
  IF
    (EXTRACT(DAYOFWEEK
      FROM
        x) = 1
      OR EXTRACT(DAYOFWEEK
      FROM
        x) = 7, TRUE, FALSE));
CREATE TEMP FUNCTION
  durationcalc(distance FLOAT64,
    speed FLOAT64)
  RETURNS INT64 AS( CAST((distance*3600)/speed AS INT64) );
WITH
  dataset AS (
  SELECT
    userID,
    CAST(tiemstamp_AEST AS DATE) AS date,
    tiemstamp_AEST as timestamp_AEST,
    distance,
    round((enhanced_speed/3600),5) AS distance_per_second,
    ROUND(enhanced_speed, 6) AS enhanced_speed,
    power,
    ROUND(power / weight,4) AS wpkg,
    ROUND(power/FTP,4) AS FTP_perct,
    cadence,
    grade,
    enhanced_altitude,
    ROUND(ascent, 4) as ascent,
    heart_rate,
    ROUND(heart_rate / maxheartrate(age),4) AS max_heart_rate_perct,
    EXTRACT(DAYOFWEEK
    FROM
      tiemstamp_AEST) AS day_of_week,
    EXTRACT(HOUR
    FROM
      tiemstamp_AEST) AS hour
  FROM
    `redbackoperationsdataai.Master_Fitness_Data.master-fitness-activity`
  -- WHERE
  --   userID = "U1000000"
  ORDER BY
    timestamp_AEST,
    distance )
SELECT
  *
FROM
  dataset
WHERE
  power IS NOT NULL
  AND cadence IS NOT NULL
  AND heart_rate >0
  AND EXTRACT(YEAR
  FROM
    timestamp_AEST) = 2022 -- 2022 only.
  AND grade IS NOT NULL
  AND enhanced_altitude IS NOT NULL
ORDER BY
  userID,
  timestamp_AEST
