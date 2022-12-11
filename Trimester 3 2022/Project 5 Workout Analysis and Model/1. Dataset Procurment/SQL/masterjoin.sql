-- MASTER FILE OPERATIONS
SELECT
  timestamp,
tiemstamp_AEST,
date_AEST
distance,
enhanced_altitude,
ascent,
grade,
calories,
enhanced_speed,
heart_rate,
temperature,
cadence,
power,
-- left_right_balance -- REMOVED DUE TO ERRORS (Garmin VS Wahoo related issue)
gps_accuracy,
sessionID,
userID,
age,
gender,
weight,
FTP,
FROM
  `redbackoperationsdataai.Fitness_Data.fitness-activity-*` -- WILDCARD
