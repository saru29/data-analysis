IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='test_pro_db' and xtype='U')
CREATE TABLE tempdb.dbo.test_pro_db
(
timestamp datetime,
	timestamp_AEST	varchar(60),
date_AEST date,
	distance float,
	enhanced_altitude float,
	ascent int,
	grade float,
	calories int,
	enhanced_speed float,
	heart_rate int,
	temperature int,
	cadence int,
	power int,
	gps_accuracy int,
	sessionID varchar(60),
	userID varchar(30),
	age int,
	gender varchar(30),
	weight int,
	FTP int
);