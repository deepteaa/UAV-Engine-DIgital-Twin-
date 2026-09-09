class DigitalTwin:

    def __init__(self):

        # The twin starts with a healthy engine

        self.expected_cht = 120


    def update(self, rpm, load):

        # Healthy engine assumptions

        cooling_efficiency = 1.0


        # Calculate heat generation

        heat_generation = (

            rpm * 0.02

            + load * 30

        )


        # Calculate cooling

        cooling_effect = (

            40 * cooling_efficiency

        )


        # Calculate target temperature

        target_cht = (

            80

            + heat_generation

            - cooling_effect

        )


        # Thermal inertia

        self.expected_cht = (

            self.expected_cht

            + (target_cht - self.expected_cht) * 0.1

        )


        return self.expected_cht