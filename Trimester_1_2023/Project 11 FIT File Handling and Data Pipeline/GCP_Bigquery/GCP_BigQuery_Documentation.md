

# GCP PROJECT - DATE AND AI

This The Google Cloud Project houses key resources and datasets which intially support sandbox data environment, crucial for various projects at Redback (DATA/AI). While Redback currently does not have a formal data warehouse (Project 18 aims to correc this) and there are challenges with the recording of bike/Wahoo Kickr data (IoT are working on this), the datasets provided here are designed to closely mirror the 'final' or actual datasets that we anticipate will be available in the future.

These datasets, which originate from a variety of sources, are instrumental for enabling the DATA/AI team to commence working on key user-centric and business-centric features. They provide a tentative structure and understanding of how our eventual data infrastructure could be set up, thus paving the way for future developments.

This sandbox environment is an important part of our strategy for data management and usage, enabling experimentation and development in a flexible, yet representative setting. It allows our team to explore and implement innovative strategies and methods, in anticipation of the future expansion of our data capabilities.

# Project Recap

- Project 11: FIT File Handling and Data Pipeline 
- Project 12: Corporate Reporting
- Project 13: The Cyclist/User Categorisation Project.
- Project 14: Sentiment analysis (language processing) and Community standards User/Community comments
- Project 15: User Ranking - Engagement
- Project 16: Performance Ranking (User)
- Project 17: Workout Categorisation
- Project 18: Data Warehouse
- Project 19: Google Analytics/Hotter Analytics/MixPanel/App Analytics (Marketing and UX)
- Project 20: Posture Analysis
<br><br>

# Access to GCP Project / BigQuery

To gain access to the GCP Project / BigQuery, please contact your team lead. They will need to coordinate with:

- Scott Blackburn (Senior Technical Officer, Cloud Computing & AI, School of Information Technology)

## Datasets Overview

1. **app_analytics**: Contains `hotjar_sample_data` for our app analytics project (Project 19).
2. **bike_store**: A sourced dataset supporting analysis such as demographics. Can assist the Corporate Reporting project (Project 12).
3. **cfpb_consumer_complaint_database**: Public dataset for sentiment analysis (Project 14). Contains `complaint_database`.
4. **fitness_data**: Rolled-up collection of all `user_data` tables into a `master_data`. Houses over 4000 hours of Wahoo data. Supports various DATA/AI team projects (Projects 13, 15, 16, 17, 18).
5. **google_analytics_sample**: Used in our app analytics project (Project 19). Contains `ga_sessions_ (366)`, 366 tables of Google Analytics sessions.
6. **sentiment**: For our sentiment analysis project (Project 14). Includes `amazon_reviews`, `imdb_dataset`, and `twitter`.
7. **thelook_ecommerce**: Public dataset potentially useful for user and product analysis. May support user engagement (Project 15), and Redback's marketplace pivot. Can enhance Project 12's scope.
8. **user_data**: Contains individual user data. Rolled up into `master_data` in `fitness_data` dataset.
9. **wahoo_connect**: Houses data from the Wahoo Kickr via the BLE Cycling Power Data Collection Python script. Supports the FIT File Handling and Data Pipeline project (Project 11).

## Viewing Workspace Resources

Access all workspace resources using the identifier `SIT-23t1-fit-data-pipe-ee8896e`. Please handle these datasets responsibly. If you need further assistance or have any questions about these resources, please contact your team lead.

# Google Cloud Storage

## Buckets

The `bucket_user_data_wahoo_fit` bucket contains original CSV fitness data such as `ExerciseData_combined_csv_user1.csv` and other larger CSV files. Note: User data is sensitive and should not be widely shared or redistributed. It's suggested to establish an `object lifecycle rule`, such as sunsetting data where it's deleted after a certain period.

Google Cloud Storage buckets are used to initialize user datasets, as the size of the CSV files make it challenging to upload as a dataset directly into BigQuery.

*Author: Mark Telley, 2023*
