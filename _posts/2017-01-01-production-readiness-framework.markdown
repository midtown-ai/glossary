---
layout: post
# layout: single
title:  "Production readiness framework"
date:   2017-01-01 12:51:28 -0800
categories: jekyll update
---

{% include links/all.md %}


# Dump ==
 ```
PRA Day Recording
Audio
Audio + Video


Production Readiness Checklist

Project Codenames & Release Dates
In the early stages of a project, codenames are often used for the various components/features used within the service.
Share the codenames used within this services these services and supply a short description or link to further documentation that will help us understand them better.

Codename

Description / Links

Expected Release Date















Service Business Requirements
Document high-level business use-cases and business requirements.



Service Description
What are the key value adds? What is the service supposed to do? What problems does it solve? Explain in a brief summary as if we were a potential customer. (Link to marketing documents are good as well)


1. Pre-Production

A. Reliability
Share high-level architecture and design of the service here. Insert diagrams, if appropriate. Address following items, if they are applicable to your service.

SR01: Full-Service description, what are the major features, at a very high level how do they work?
SR02: Full-Service workflow diagram (at a very high level what components are involved and how do they talk to one-another API/HTTPS/etc..)
SR03: Full-Service Network topology
SR04: Full Service Storage architecture and design
SR05: Full-Service global data flow (include data types, encryption types, etc.)
SR06: Database and persistence
SR07: Toggled features – Is there any feature that are turned off (A / B rollout of features, limited testing or use of features, etc.)?
SR08: Is there any 'customer sensitive' data coming from or going to 3rd party SaaS services? (I.e. customer names, IP address, Usernames, Password, SSN's etc)
SR09: Is anything being stored internationally (in a country different to where the service is located)?
SR10: How are changes to the service controlled and monitored? Do you check for configuration drift?
SR11: Are configuration changes auditable – who made the change when?
SR12: Is this product being dogfooded if so, by who and what is the feedback?
SR13: Is the application(s) resilient to component failure or rollbacks? If so, how does the application handle these failures? (e.g. Database is rolled back 5 minutes due to BC/DR event, how does the application handle this roll back / loss of committed transactions?)



B. Security
vSecure Dashboard (PLEASE CHANGE to reflect the right code name - ask Tom Ralph if you need assistance with this)




 20 Jul 2017

Page not found for multiexcerpt macro.
The page: InfoSec:Cloud Service Security Status - Goldilocks was not found. Please check/update the page name used in the 'multiexcerpt-include macro.
Page not found for multiexcerpt macro.
The page: InfoSec:Cloud Service Security Status - Goldilocks was not found. Please check/update the page name used in the 'multiexcerpt-include macro.


Page not found for multiexcerpt macro.
The page: InfoSec:Cloud Service Security Status - Goldilocks was not found. Please check/update the page name used in the 'multiexcerpt-include macro.
Data Source Link: <CHANGEME>



SS01: Does this service need to comply with any Information Security Regulations? (e.g., HIPAA, PCI-DSS)
SS02: How regularly is security testing conducted?
SS03: Do you use certificates? If yes, how do you manage them?
SS04: How are you protecting access to and use of the root/local administrator account credentials?
SS05: Which Identity Management solution is used to manage user/service accounts?
SS06: How are you defining the Global Password Policy for the service?
SS07: How are you defining roles and responsibilities of system users to control human access to the Management Interfaces and API?
SS08: How are you restricting automated access to resources? (e.g., applications, scripts, and/or third-party tools or services)
SS09: How are you testing for security vulnerabilities on the platform?
SS10: How is malicious activity and security policy violations being monitored?
SS11: How are you capturing and analyzing system logs?
SS12: What logs are not sent in Syslog or central logging?
SS13: What is your syslog file retention policy?
SS14: How are you enforcing network and host-level boundary protection?
SS15: What security zones exist and how are you restricting communication between the security zones (i.e., how do you prevent the web tier from accessing the database tier)?
SS16: How are you protecting the integrity of the operating systems within your virtual machines/containers?
SS17: What is the security patching/update policy for the platform components?
SS18: How are you classifying your data and classifications are being defined?
SS19: How are you encrypting or protecting your data at rest?
SS20: How are you encrypting or protecting your data in transit?
SS21: How are you managing encryption keys (if required)?
SS22: How are security incidents managed?
SS23: How often do you plan on reviewing your security measures and procedures?
SS24: How are your accounts and credentials segregated? Provide a list of all accounts and group definitions for each individual component of the service.

C. Performance (API / Code performance)
SP01: What are the Performance SLA/OLA for this service? (UI/API repose times etc)
SP02: What Key Performance Indicators (KPI) are being measured? Metrics are just one part of a KPI, how is the service gaining insight into itself with the metrics being measured?
SP03: How do you monitor resources to ensure they are performing as expected?
SP04: What are the consequences If the Performance SLA is missed?
SP05: What metrics were used to size the platform?
SP06: What is the Demand/Capacity Management Policies in place?
SP07: How does your system adapt to changes in demand, Manual process or automated?
SP08: How is the platform tested to ensure it can meet the Performance SLA's?
SP09: What is the [expected/estimated] request volume for each component of your system? What do you think your ramp is? What is the expected per-customer performance?
SP10: What default resource limits (soft and hard) are you imposing on customers?
SP11: How are you doing throttling in each component (customer and component-to-component) of your service?
SP12: How do you handle brief spikes in load? How much additional spike can your service support, and for how long?
SP13: What is the customer experience for getting limits and throttles raised? How do you raise a limit or throttle on your side?
SP14: Are there any potential bottlenecks in the design that could affect the performance of the service?

D. Compliance
SC01: Service compliance and legal requirements to be added here.
SC02: If an SLA, either Availability or Performance SLA/OLA is not met, how is the customer both notified and compensated for this breach of contract?
SC03: Where are the source code and key intellectual property stored?

SC04: Please provide a diagram of where data flows, how it flows (encrypted or not), what the data is (PII, logs, etc.), etc

SC05: What are your future (12 months) expansion plans? (New compliance requirements/regions?)

2. Deployment
Describe your deployment procedure(s). The deployment process is expected to be fully automated and documented. Share your deployment documentation before the assessment review, and address following items, if they are applicable to your service.
DP01: What are the manual touch points in your system? Partial deployment (canary, staging, and A/B testing deployments), Full deployment and Rollback
DP02: Are there any external dependencies that need to be in place before deployment can begin? If so, how are the pre-requisites put in place and by whom?
DP03: What is the process for adding additional physical resources to cope with increased demand?
DP04: Is the deployment method documented? If so, where is the documentation published?


3. Production
A. Business Continuity
Service Availability (along with Service Level Agreement (SLA)), Disaster Recovery (DR) – along with Recoverability is covered in this section.
Address following items, if they are applicable to your service.


BC01: Service SLA – number of 9's (e.g. 3 nines (99.9%), 3 ½ nines (99.95%), etc.)
BC02: DR design and implementation – along with Recovery Point Objective (RPO) and Recovery Time Objective (RTO) targets.
BC03: GameDay Plans (including plan for (planned) periodic failover)
BC04: Is the service split across physical sites/availability zones, if so, how? What happens when service between physical sites/availability zones fails?
BC05: Single points of failures in the service.
BC06: External dependencies and how service will be impacted based on a failure of each of the dependencies. Ensure to include supporting services (e.g. monitoring, DDNS, billing, CI/CD, etc.)

External Dependency

SLA/OLS of Dependency

Impact of Dependency Failure















BC07: Backup/Restore and data integrity of backup system (checksums, metadata, etc.)
BC08: How are the backups of stateful components handled? Is it required for all stateful components to be backed up at the same time? Or is there a 'bring forward' function to catch up a restored component?
BC09: Provide a timeline for a global presence. Will each location be an isolated instance or a single global instance?



B. Monitoring (How is the code running in production? Is the service up/down?)
PM01: What are you alarming on? What are the thresholds? List all of your monitors with period and threshold, and sev2/sev2.5/sev3.
PM02: Include a link to your main dashboards. When do you look at your dashboards?
PM03: Are there metrics you know you don't have instrumented? Metrics that you have on your dashboards but are not suitable for alarming?
PM04: Do you have escalation thresholds defined? List those thresholds.
PM05: How do you measure an end to end customer experience?
PM06: How do you trace a customer request at any point in your system?
PM07: How does your canary function? What does it exercise? Is it synchronous, asynchronous? Do you have a heads-up view of it? How often does it kick off, how often does it finish? Does it alarm on a failure in the middle of an asynchronous run, or does it need to finish? When does it alarm? How do you manage and deploy it?
PM08: How are you metering customer usage?
PM09: How are you monitoring physical resources? (e.g., CPU/MEM/HDD)
PM10: Attach diagrams of how your components (including canary, backups, and operational tools) are arranged logically, including dependencies. (Use multiple diagrams depending on the number of components and how many dependencies each has.)
PM11: How does the service verify the monitoring solution is online?

C. Telemetry (Business Telemetry, what features are being used?)
Service usage, and ability to perform synthetic operations in stage and production environments are covered here. Please include links to VAC and Tableau for reports that are used by the service.


PP01: Telemetry coverage for the service
PP02: Testing in Production (TiP) – Support for synthetic operations in stage and production deployments
PP03: Provide links to the products VAC/Tableau page(s)



D. Utilization and Cost Optimization
Efficient utilization of production capacity is part of VMware's Operational Excellence goals. Address following items, if they are applicable to your service.

PU01: Is cost a parameter when selecting the compute/network/storage components for the service?
PU02: Right-sizing: How do you make sure your capacity matches but does not substantially exceed what you need? How did you determine the capacity required for the launch?
PU03: Have you sized your resources to meet your cost targets?
PU04: Have you selected the appropriate pricing model to meet your cost targets?
PU05: Did you consider data-transfer charges when designing your architecture?
PU06: How are you monitoring usage and spending?
PU07: Do you decommission resources that you no longer need, or stop resources that are temporarily not needed? If so, is this process automated?
PU08: How regularly are you auditing service usage to determine if more or fewer resources are required?
PU09: What access controls and procedures do you have in place to govern cloud service usage (if appropriate)?
PU10: What areas of the service could be optimized to get a better return on investment (ROI)?

E. Engagement and Escalation
Central SRE team will be the first line of defense for your service. This section covers areas that help to establish an efficient bridging mechanism between your engineering team and your customers.
Address following items, if they are applicable to your service.


PE01: How does your team's on-call rotation look (link to on-call)? How do you do hand-off?
PE02: Who on your team has access to each IAM User and Role? Additionally, who on your team has can access and configure other components of your prod service (e.g. databases, sudo, pipelines)? What criteria do members of your team have to meet before access to AWS accounts and other components of your prod service and is granted?
PE03: Who has access to your RBAC portal?
PE04: Who on our team has can access and configure production service (e.g. databases, sudo, pipelines)?
PE05: What criteria do members of your team have to meet before access to accounts and other components of your production service is granted?
PE07: Link to your runbooks. What known gaps are there in your runbooks?
PE08: When is your weekly ops meeting? What's the agenda? Who attends?
PE09: How and when will your team escalate to managers? Are the managers comfortable engaging, running calls, and doing basic troubleshooting?
PE10: What operational scripts or tools do you have in place to debug customer issues or determine customer impact?
PE11: Please provide a table listing all customer-facing API operations, an explanation for what each does, and the components and dependencies of your service that it touches.
PE12: What does the lifecycle of each customer request look like across your control and data planes? Chart each operation for your service on a data flow diagram that represents every service component and dependency. Describe the credentials used across the lifecycle of each operation--are you making calls on behalf of customers?
PE13: How are your Operations/Support teams trained to be able to manage and support this service?
PE14: What is your escalation process in the event of a Security issue?
PE15: After a PO or P1 outage, what postmortem processes do you have in place to ensure you don't encounter repeat issues going forward? (attach a Postmortem report if available)

PE16: Does the service follow the recommended slack channels (Slack Service Communication Standards) ?



Potential Operational Risks
What are the top three things that aren't on fire yet, but will be?

...
...
...
Greatest Accomplishments
What are the top three things you are most proud of in the service? (Would you be willing to share these are the Bi-Weekly PRA Service Review meeting)

...
...
...
Biggest Lesson Learned:
What are the top three things that the service has learned, that we can share with other teams?

...
...
...


Remaining Actions
Enumerate your remaining actions and correction dates.



APPENDIX A: Notification and Communication Process

<TO BE COMPLETED>
Customer Communications Plan
Incident Management Communications
Maintenance Communications

APPENDIX B: Service Health KPI Best Practices
Common SaaS KPI Objective
Define, Validate and Institutionalize the key SaaS KPI’s across VMWare to Measure and Monitor performance of new and evolving SaaS Offerings in a Consistent and Repeatable manner

KPI Name

KPI Topic

(Proposed) Target

Note

Availability

Service Console

99.9% monthly

Availability - A KPI given in a numeric percent (xx.xxxx %)
Availability is usually expressed as a percentage of uptime in a given month or year.

 Definition of availability may change from one service to another. Portal (front-end) availability may be separated from back-end availability, as long as there is programmatic access to the service when the portal is down.

Services are not expected to have planned downtime.



Expected SLA targets:

Three nines (99.9% = 43.8 min/month = 8.76 hours/year)

Three and a half nines (99.95% = 21.56 min/month, 4.38 hours/year)

Service APIs

Response time < 2 sec

Latency < 100 ms


Response

Mean Time to Acknowledge (MTTA) P0/P1

15 minutes


Mean Time To Relief (MTTR) P0/P1

Tracking KPI


Business Continuity

Recovery Point Objective (RPO)

Tracking KPI

The maximum targeted period in which data might be lost from a service due to an incident leading to database failure

Recovery time objective (RTO)

Tracking KPI

The targeted duration of time within which a service must be restored after a Service Outage

Quality of Service

Number of P0 incidents

Tracking KPI


Monitoring

Percentage of P0 incidents caught by monitoring

Tracking KPI


Security

Number of P0 security incidents

Tracking KPI

 ```

# Sample reports ==
## Telemetry ===
 ```
We can measure a few items today through logs, surveys, ticketing of P0/P1.  A few measurements we could
measure includes:
On-boarding time - The amount of time it takes a customer to onboard the service to when they can use the service.
Support Response time - The speed of which we respond to a support request or service ticket.
           Slack #sm-support
           Emails: TBD (goal 4 hours)
           Live chat: TBD (goal 1 minute)
Number of customer complaints
Number of P0/P1tickets filed per week/month/qtr.
Satisfaction surveys - product and service satisfaction
API response time
Click rates (google analytics)
Recurring Revenue
Attrition Rate (churn)
Average Recurring Revenue (avg. sales price)
Customer Acquisition Cost (per customer)
 ```
 ```
Area
Metric

Definition

Measurement
Beta Target

Prod Target
Service

On-boarding time

The amount of time it takes a customer to onboard the service to when they can use the service.


Non-contact rate

The percentage of customers who encountered big problems but either 1) did not bother to contact you or 2) tried, and eventually gave up. A high non-contact rate could signify that it is difficult for customers to contact you.


Support Response time

The speed of which we respond to a support request or service ticket.

Slack #sm-support
Emails: TBD (goal 4 hours)
Live chat: TBD (goal 1 minute)


Number of customer complaints (P0/P1)



Number of tickets filed per week/month/qtr.


Satisfaction surveys

 These measures of customer satisfaction should be performed consistently throughout the year with a simple one or two question survey, such as a basic rating scale and an open-ended text field where customers can explain why they gave that rating (for product and support)


Percent abandoning contact attempt

This includes all interactions that customers might abandon, such as social media conversations, phone calls, live chat, and email. Similar to high non-contact rates, a high contact abandonment rate could mean your current service organization is not equipped to deliver quick, effective service.

API response time

Recurring Revenue


Attrition Rate (churn)

Average Recurring Revenue (avg. sales price)

Customer Acquisition Cost (per customer)

Customer Lifetime Value
 ```

