  -- -- Author: MTELLEY
  -- Release 1
  -- CLEANSE AND ORGANISE DATA FROM CSV FILE
  -- TO DO: Create SessionID using rank/row ops (Version 2)
  -- //////////////////////////////////////////
SELECT
  -- WORKOUT METRICS
  timestamp,
  TIMESTAMP(DATETIME(CAST(CONCAT(SPLIT(timestamp, '+')[SAFE_ORDINAL(1)]," ","UTC") AS TIMESTAMP), "Australia/Melbourne")) AS tiemstamp_AEST,
  CAST((TIMESTAMP(DATETIME(CAST(CONCAT(SPLIT(timestamp, '+')[SAFE_ORDINAL(1)]," ","UTC") AS TIMESTAMP), "Australia/Melbourne"))) AS DATE) AS date_AEST,
  -- position_lat, -- removed for privacy reasons
  -- position_long, -- removed for privacy reasons
  SAFE_CAST(distance AS FLOAT64) AS distance,
  SAFE_CAST(enhanced_altitude AS FLOAT64) AS enhanced_altitude,
  -- altitude, -- removed
  SAFE_CAST(ascent AS FLOAT64) AS ascent,
  SAFE_CAST(grade AS FLOAT64) AS grade,
  SAFE_CAST(calories AS FLOAT64) AS calories,
  SAFE_CAST(enhanced_speed AS FLOAT64) AS enhanced_speed,
  -- speed, -- removed
  SAFE_CAST(heart_rate AS FLOAT64) AS heart_rate,
  SAFE_CAST(temperature AS INT64) AS temperature,
  SAFE_CAST(cadence AS FLOAT64) AS cadence,
  SAFE_CAST(power AS FLOAT64) AS power,
  ROUND(SAFE_CAST(left_right_balance AS FLOAT64)/100,2) AS left_right_balance,
  SAFE_CAST(gps_accuracy AS FLOAT64) AS gps_accuracy,
  -- PRODUCT DETAILS:
  -- descriptor,
  -- product_name,
  -- serial_number, -- removed for privacy reasons
  sessionID,
  -- TO DO
  CASE
    WHEN Timestamp IS NOT NULL THEN "U1000000" -- USER ID, MANUALLY SET
  ELSE
  NULL
END
  AS userID,
  CASE
   WHEN Timestamp IS NOT NULL THEN 33 -- AGE, MANUALLY SET
  ELSE
  NULL
END
  AS age,
  CASE
    WHEN Timestamp IS NOT NULL THEN "MALE" -- GENDER, MANUALLY SET
  ELSE
  NULL
END
  AS gender,
  CASE
    WHEN Timestamp IS NOT NULL THEN 80 -- WEIGHT, MANUALLY SET
  ELSE
  NULL
END
  AS weight,
  CASE
    WHEN Timestamp IS NOT NULL THEN 301 -- FTP, MANUALLY SET
  ELSE
  NULL
END
  AS FTP,
FROM
  `redbackoperationsdataai.Fitness_Data.fitness-activity-user1` -- UPDATE TO CORREC TABLE NMAE
WHERE
  position_lat IS NOT NULL -- removes not workout related data
  AND distance IS NOT NULL
  AND timestamp != "timestamp" -- remove header column
ORDER BY
  timestamp ASC