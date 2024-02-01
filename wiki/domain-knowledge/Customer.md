# Customer and Customer Hierarchies 
The solution we are building is meant for manufacturers. So from this perspective the retailers are our customers.
Customers are represented as a hierarchy. The hierarchy is a tree structure that represents the relationships between customers, where customers represent the nodes.

## Customer 
Customers are usually identified by a unique identifier, and have a name and a type. The type of a customer can be a retailer, a distributor, a wholesaler, etc.


## Customers in the context of promotions
Customers are used in promotions to define the target audience of the promotion. A promotion can be targeted to a specific customer, a group of customers, or all customers. The target audience of a promotion is defined by the customer hierarchy. 
If a proof of performance is send for a specific store, it could be that the promotion is linked to a customer further up in the hierarchy and we need to traverse the hierarchy to find the right promotion. 

E.g. Red bull negotiated a promotion with Walmart to promote their energy drink. The promotion is targeted to all Walmart stores in California. The proof of performance is send for a specific store in Irvine. We need to traverse up the customer hierarchy to find the right promotion.

## Example Hierarchy: 
- Walmart US 
  - Walmart US - West
    - Walmart US - West - California
      - Walmart US - West - California - Los Angeles
        - Walmart US - West - California - Los Angeles - Store 1
        - Walmart US - West - California - Los Angeles - Store 2
        - ...
    - Walmart US - West - Nevada
    - ...
- Albertsons
  - Albertsons - West
    - Albertsons - West - California
      - Albertsons - West - California - Los Angeles
        - Albertsons - West - California - Los Angeles - Store 1
        - Albertsons - West - California - Los Angeles - Store 2
        - ...
    - Albertsons - West - Nevada
    - ...
