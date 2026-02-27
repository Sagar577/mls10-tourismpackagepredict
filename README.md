## Problem Statement - Tourism Package Prediction

**Context**
"Visit with Us," a leading travel company, is revolutionizing the tourism industry by leveraging data-driven strategies to optimize operations and customer engagement. While introducing a new package offering, such as the Wellness Tourism Package, the company faces challenges in targeting the right customers efficiently. The manual approach to identifying potential customers is inconsistent, time-consuming, and prone to errors, leading to missed opportunities and suboptimal campaign performance.

To address these issues, the company aims to implement a scalable and automated system that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies. By utilizing an MLOps pipeline, the company seeks to achieve seamless integration of data preprocessing, model development, deployment, and CI/CD practices for continuous improvement. This system will ensure efficient targeting of customers, timely updates to the predictive model, and adaptation to evolving customer behaviors, ultimately driving growth and customer satisfaction.


**Objective**
As an MLOps Engineer at "Visit with Us," your responsibility is to design and deploy an MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer purchases. The primary objective is to build a model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them. The pipeline will include data cleaning, preprocessing, transformation, model building, training, evaluation, and deployment, ensuring consistent performance and scalability. By leveraging GitHub Actions for CI/CD integration, the system will enable automated updates, streamline model deployment, and improve operational efficiency. This robust predictive solution will empower policymakers to make data-driven decisions, enhance marketing strategies, and effectively target potential customers, thereby driving customer acquisition and business growth.


**Data Disctionary**
The dataset contains customer and interaction data that serve as key attributes for predicting the likelihood of purchasing the Wellness Tourism Package. The detailed attributes are:

**Customer Details**


1. CustomerID:Unique identifier for each customer.
2. ProdTaken:Target variable indicating whether the customer has purchased a package (0: No, 1: Yes).
3. Age:Age of the customer.
4. TypeofContact:The method by which the customer was contacted (Company Invited or Self Inquiry).
5. CityTier:The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
6. Occupation:Customer's occupation (e.g., Salaried, Freelancer).
7. Gender:Gender of the customer (Male, Female).
8. NumberOfPersonVisiting:Total number of people accompanying the customer on the trip.
9. PreferredPropertyStar:Preferred hotel rating by the customer.
10. MaritalStatus:Marital status of the customer (Single, Married, Divorced).
11. NumberOfTrips:Average number of trips the customer takes annually.
12. Passport:Whether the customer holds a valid passport (0: No, 1: Yes).
13. OwnCar:Whether the customer owns a car (0: No, 1: Yes).
14. NumberOfChildrenVisiting:Number of children below age 5 accompanying the customer.
15. Designation:Customer's designation in their current organization.
16. MonthlyIncome:Gross monthly income of the customer.

**Customer Interaction Data**

17. PitchSatisfactionScore:Score indicating the customer's satisfaction with the sales pitch.
18. ProductPitched:The type of product pitched to the customer.
19. NumberOfFollowups:Total number of follow-ups by the salesperson after the sales pitch.
20. DurationOfPitch:Duration of the sales pitch delivered to the customer.


