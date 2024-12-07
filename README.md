# David Silver's Car Rental Problem

![David Silver's Car Rental Problem](static/silver-1.jpeg)

- States: Two locations, maximum 20 cars per location
- Actions: Move cars from one location to another, maximum 5 cars per move
    - Action of 5 results in 5 cars moved from location 1 to location 2.
- Rewards: $10 for each car rented, only if the car is available
- Transitions: Car returned and requested are sampled from Poisson distributions:
    - Poisson distribution, n returns ($\lambda_r$) or requests ($\lambda_p$) with prob $\lambda^n e^{-\lambda} / n!$
    - Returns:
        - Location 1: $\lambda_r = 3$
        - Location 2: $\lambda_r = 2$
    - Requests:
        - Location 1: $\lambda_p = 3$
        - Location 2: $\lambda_p = 4$
    - Cost: $0 per car moved


