# Introduction

## Industry | B2C Healthcare Insurance

The B2C (business-to-consumer) healthcare insurance industry encompasses companies offering individual and family health insurance plans directly to consumers. These plans provide coverage for medical expenses incurred for various health services and treatments. Key features of the B2C healthcare insurance industry include:

* Focus on individual and family plans: Unlike B2B (business-to-business) insurance, which caters to employer-sponsored plans, B2C primarily focuses on individual and family policies purchased directly by consumers.
* Diverse product offerings: Companies offer a variety of health insurance plans with different coverage levels, deductibles, co-pays, and network options to cater to diverse needs and budgets.
* Competitive landscape: The market is highly competitive, with numerous insurance providers vying for market share. This competition can lead to innovative product offerings and competitive pricing.
* Regulatory environment: The industry is heavily regulated by government agencies to ensure consumer protection and financial stability.
* Technology adoption: Technology plays a significant role in the industry, with online enrollment platforms, telemedicine services, and mobile apps becoming increasingly popular.

## Business Challenge | Marketing Mix Optimization

Commited to maximizing [ROAS](https://www.apexure.com/blog/roi-vs-roas-which-is-the-better-metric-for-ad-campaigns) within their $70 million+ annual budget, the marketing team of the nation's largest supplemental insurance provider actively sought recommendations to optimize lead generation and conversions for their insurance products.

Historically, the organization had mainly focused on traditional approaches based on brand marketing to address their core objectives of driving website and call center leads to generate new healthcare premiums. However, this approach lacked a cross-channel view of marketing spend distribution and performance.

The design and adoption of a "test-and-learn" framework that focused on incremental and continous improvements to marketing spend distribution informed via experimenation was needed. One of the first steps identified to help inform and guide this approach was the development of a marketing mix model (MMM).  The model developed served as a catalyst and "North Star" for the marketing organization's adoption of a more data-driven approach to marketing spend optimization. 

The following case study provides a technical summary for the development of the marketing mix model along with some of the next-step recommendations coming out of that exercise.

## Analysis Technique | Marketing Mix Model (MMM)

**Marketing Mix Model (MMM)**<br>
Marketing Mix Modeling (MMM) is a data-driven approach that helps businesses understand how different marketing activities and channels contribute to their key goals, such as sales, conversions, or brand awareness. Overall, MMM is a powerful tool that can help businesses maximize the return on their marketing investment by providing data-driven insights and optimizing their marketing efforts.

**The Evolution of MMM**<br>
While MMM has been a longstanding tool for measuring marketing effectiveness, with the advent of digital marketing [MTA (Multi-Touch Attribution)](https://en.wikipedia.org/wiki/Attribution_(marketing)) gained prominence by providing a more detailed attribution approach. MTA enabled the monitoring of individual users across various touchpoints, allowing marketers to pinpoint specific influencers on conversions and allocate budget accordingly. This precision made MTA especially popular among digital marketers.

However, new privacy regulations like [CCPA](https://oag.ca.gov/privacy/ccpa/regs) and [GDPR](https://gdpr.eu/) are posing challenges to MTA's cookie-based tracking. As a result, MMM, which relies on aggregate data rather than user-level data, is experiencing a resurgence in relevance.

## MMM Libraries | Robyn and LightweightMMM

At the time of the original analysis, the model was developed using a multiple linear regression (MLR) technique.  However, since then several open-source libraries have been developed for MMM specifically.  For the development of this case study, I assessed two of these libraries developed by major technology companies:

* [Robyn (Meta)](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/) - An open-source project for marketing mix modeling (MMM) written in R developed by Meta's Marketing Science division.
* [LightweightMMM (Google)](https://lightweight-mmm.readthedocs.io/en/latest/) - An open-source library written in Python developed and maintained by a team of Google engineers primarily focused on improving marketing mix modeling (MMM) techniques. While not an official Google product, it leverages Google's internal expertise and research in the field of marketing analytics.

I decided to go with LightweightMMM, mainly due to my familiarity with Python and the [Bayesian approach](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/) which underpins the library and I was interested in learning more about.


