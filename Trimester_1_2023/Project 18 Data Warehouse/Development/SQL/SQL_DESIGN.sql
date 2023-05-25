USE [master_data]
GO

/****** Object:  Table [dbo].[md_dataload]    Script Date: 5/26/2023 9:13:53 AM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[md_dataload](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[table_name] [varchar](30) NULL,
	[start_date] [date] NULL,
	[end_date] [date] NULL,
	[active_flag] [int] NOT NULL,
	[created_on] [datetime] NULL,
	[created_by] [varchar](15) NULL,
	[modified_on] [datetime] NULL,
	[modified_by] [varchar](15) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO

ALTER TABLE [dbo].[md_dataload] ADD  DEFAULT (getdate()) FOR [created_on]
GO

ALTER TABLE [dbo].[md_dataload] ADD  DEFAULT ('PSAPKOTA') FOR [created_by]
GO

ALTER TABLE [dbo].[md_dataload]  WITH CHECK ADD CHECK  (([active_flag]=(1) OR [active_flag]=(0)))
GO