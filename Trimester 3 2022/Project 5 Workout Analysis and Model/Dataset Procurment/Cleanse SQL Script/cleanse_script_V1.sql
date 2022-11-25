/*
 *   Copyright (c) 2022 Author: MARK TELLEY. Student ID: 220533036 
 *   All rights reserved.
 */
 
-- Author: MTELLEY
-- DRAFT V1
-- CLEANSE AND ORGANISE DATA FROM CSV FILE
-- TO DO: Create SessionID using rank/row ops
-- //////////////////////////////////////////
SELECT
  -- WORKOUT METRICS
  timestamp,
  TIMESTAMP(DATETIME(CAST(CONCAT(SPLIT(timestamp, '+')[SAFE_ORDINAL(1)]," ","UTC") AS TIMESTAMP), "Australia/Melbourne")) AS tiemstamp_AEST,
  CAST((TIMESTAMP(DATETIME(CAST(CONCAT(SPLIT(timestamp, '+')[SAFE_ORDINAL(1)]," ","UTC") AS TIMESTAMP), "Australia/Melbourne"))) AS DATE) as date_AEST,
  position_lat,
  position_long,
  SAFE_CAST(distance AS FLOAT64) as distance,
  SAFE_CAST(enhanced_altitude AS FLOAT64) as enhanced_altitude,
  -- altitude, -- removed
  SAFE_CAST(ascent AS FLOAT64) as ascent,
  SAFE_CAST(grade AS FLOAT64) as grade,
  SAFE_CAST(calories AS FLOAT64) as calories,
  SAFE_CAST(enhanced_speed AS FLOAT64) as enhanced_speed,
  -- speed, -- removed 
  SAFE_CAST(heart_rate AS FLOAT64) as heart_rate,
  SAFE_CAST(temperature AS INT64) as temperature,
  SAFE_CAST(cadence AS FLOAT64) AS cadence,
  SAFE_CAST(power AS FLOAT64) as power,
  SAFE_CAST(left_right_balance AS FLOAT64)/100 as left_right_balance,
  SAFE_CAST(gps_accuracy AS FLOAT64) as gps_accuracy,
  -- PRODUCT DETAILS:
  -- descriptor,
  -- product_name, 
  -- serial_number,
  sessionID
FROM
  `heyrcg-dataanalytics.RedbackOpeartionsMTelley.fitness_activity_mtelleyV2`
WHERE
  position_lat IS NOT NULL -- removes not workout related data
  AND timestamp != "timestamp" -- remove header column
  -- AND ascent is NOT NULL
  AND CAST((TIMESTAMP(DATETIME(CAST(CONCAT(SPLIT(timestamp, '+')[SAFE_ORDINAL(1)]," ","UTC") AS TIMESTAMP), "Australia/Melbourne"))) AS DATE) = "2022-11-18"
ORDER BY
  timestamp ASC
-- LIMIT 10