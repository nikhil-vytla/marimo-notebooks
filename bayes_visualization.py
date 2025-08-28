import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(mo):
    mo.md(
        r"""
    # An Interactive Look at Bayes' Theorem

    This notebook provides an interactive exploration of Bayes' theorem, inspired by [this visualization](https://setosa.io/ev/conditional-probability/) and [this article](https://oscarbonilla.com/2009/05/visualizing-bayes-theorem/).

    ## Definitions 
    First, let's define a few terms:

    /// admonition | Conditional probability:

    The probability of an event $A$ occurring given another event $B$ has already occurred.

    $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

    where $P(B) \neq 0$.
    ///

    Conditional probabilities let us answer questions like:

    > "What's the probability a person with symptoms of the Flu actually has the Flu?"

    Statistically speaking, this is asking: $P(\textcolor{blue}{\text{Flu}} | \textcolor{red}{\text{Symptoms}})$. In order to answer this, we need to directly observe or measure this with data.

    /// admonition | Bayes' Theorem:

    A mathematical rule for *inverting* conditional probabilities.

    $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

    where $A$ and $B$ are events and $P(B) \neq 0$.
    ///

    In the land of Bayes:

    - $P(A|B)$: the probability of $A$ occurring given that $B$ is true, also known as the **posterior probability** of $A$ given $B$.
    - $P(B|A)$: the probability of $B$ occurring given that $A$ is true, also known as the **likelihood of** of $A$ given (a fixed) $B$.
    - $P(A)$ and $P(B)$: the probability of observing $A$ and $B$ respectively without any conditions, also know as the **prior probability** and **marginal probability**, respectively.

    /// details | Aside: a mini derivation of Bayes' theorem
    Given the above definition of conditional probability, we can similarily state that for events $B$ and $A$:

    $$P(B|A) = \frac{P(A \cap B)}{P(A)}$$ 

    if $P(A) \neq 0$.

    Thus, solving for $P(A \cap B)$ and substituting the value into the expression for $P(A|B)$, we get Bayes' theorem:

    $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

    if $P(B) \neq 0$.

    ///

    With Bayes' theorem, we can answer questions like:

    > "A person tests positive for the Flu. What's the probability they actually have the Flu?"

    Statistically speaking, this is asking: $P(\textcolor{blue}{\text{Flu}} | \textcolor{red}{\text{Positive Test}})$. 

    But in this scenario, we don't directly know this (and it can be very intensive to measure), so we need to calculate it from what we do know! Let's say we know/can measure the following:

    - $P(\textcolor{red}{\text{Positive Test}} | \textcolor{blue}{\text{Flu}})$ - **test sensitivity**
    - $P(\textcolor{blue}{\text{Flu}})$ - **disease prevalence**
    - $P(\textcolor{red}{\text{Positive Test}} | \textcolor{magenta}{\text{No Flu}})$ - **false positive rate (FPR)**

    We can then calculate the alternate conditional probability!

    ## Why is Bayes' theorem useful?

    In practice, it's much easier to measure test performance in controlled studies ([sensitivity/specificity](https://www.youtube.com/watch?v=vP06aMoz4v8)) than to directly measure "what % of positive tests are correct" in the real world. Bayes' theorem lets us flip the conditional relationship using data we can actually collect!
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    return alt, mo, pd


@app.cell
def _(mo):
    # Bayes Theorem Parameters

    prior_a_slider = mo.ui.slider(
        start=0.01, stop=0.99, step=0.01, value=0.3,
        label="Prior $P(A)$: Base rate of event $A$"
    )

    likelihood_slider = mo.ui.slider(
        start=0.01, stop=0.99, step=0.01, value=0.8,
        label="Likelihood $P(B|A)$: Probability of $B$ given $A$"
    )

    false_positive_slider = mo.ui.slider(
        start=0.01, stop=0.99, step=0.01, value=0.1,
        label="False Positive $P(B|¬A)$: Probability of $B$ given not $A$"
    )

    mo.md("""
    ## Parameters

    **Try adjusting the sliders to see how:**

    - Higher prior probability $P(A)$ increases posterior probability $P(A|B)$
    - Higher likelihood $P(B|A)$ increases posterior probability $P(A|B)$
    - Higher false positive rate $P(B|¬A)$ decreases posterior probability $P(A|B)$

          """)
    return false_positive_slider, likelihood_slider, prior_a_slider


@app.cell
def _(false_positive_slider, likelihood_slider, mo, prior_a_slider):
    mo.hstack([prior_a_slider, likelihood_slider, false_positive_slider])
    return


@app.cell
def _(false_positive_slider, likelihood_slider, prior_a_slider):
    # Calculate Bayes Theorem components
    prior_a = prior_a_slider.value
    prior_not_a = 1 - prior_a
    likelihood_b_given_a = likelihood_slider.value
    likelihood_b_given_not_a = false_positive_slider.value

    # Total probability of B (marginal probability)
    marginal_b = likelihood_b_given_a * prior_a + likelihood_b_given_not_a * prior_not_a

    # Posterior probability using Bayes theorem
    posterior_a_given_b = (likelihood_b_given_a * prior_a) / marginal_b

    # For visualization
    true_positive = likelihood_b_given_a * prior_a
    false_positive = likelihood_b_given_not_a * prior_not_a
    true_negative = (1 - likelihood_b_given_not_a) * prior_not_a
    false_negative = (1 - likelihood_b_given_a) * prior_a

    calculations = {
        'prior_a': prior_a,
        'prior_not_a': prior_not_a,
        'likelihood_b_given_a': likelihood_b_given_a,
        'likelihood_b_given_not_a': likelihood_b_given_not_a,
        'marginal_b': marginal_b,
        'posterior_a_given_b': posterior_a_given_b,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'true_negative': true_negative,
        'false_negative': false_negative
    }

    calculations
    return (calculations,)


@app.cell
def _(calculations, mo):
    # Mathematical breakdown
    calc = calculations

    mo.md(f"""
    ## Mathematical Breakdown

    **Given:**

    - Prior probability $P(A)$ = **{calc['prior_a']:.3f}**
    - Likelihood $P(B|A)$ = **{calc['likelihood_b_given_a']:.3f}**
    - False positive rate $P(B|¬A)$ = **{calc['likelihood_b_given_not_a']:.3f}**

    **Calculated:**

    - Marginal probability $P(B)$ = $P(B|A)×P(A)$ + $P(B|¬A)×P(¬A)$ = **{calc['marginal_b']:.3f}**
    - Posterior probability $P(A|B)$ = $P(B|A)×P(A) / P(B)$ = **{calc['posterior_a_given_b']:.3f}**

    **Interpretation:**
    The probability that $A$ is true given we observed $B$ is **{calc['posterior_a_given_b']:.1%}**.
    This is {'higher' if calc['posterior_a_given_b'] > calc['prior_a'] else 'lower'} than the prior probability of **{calc['prior_a']:.1%}**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Visualizing Conditional Probabilities and Bayes' Theorem""")
    return


@app.cell
def _(mo):
    mo.md(r"""1. **Population Distribution**: Shows the base rates (prior probabilities) of events $A$ and $¬A$""")
    return


@app.cell
def _(alt, calculations, mo, pd):
    # Prior distribution (pie chart)
    prior_data = pd.DataFrame({
        'category': ['A', '¬A'],
        'probability': [calculations['prior_a'], calculations['prior_not_a']],
        'color': ['#FF6B6B', '#4ECDC4']
    })

    prior_chart = alt.Chart(prior_data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="probability", type="quantitative"),
        color=alt.Color(field="color", type="nominal", scale=None),
        tooltip=['category', 'probability']
    ).properties(
        title="Population Distribution (Prior)",
        width=300,
        height=300
    )

    mo.ui.altair_chart(prior_chart)
    return


@app.cell
def _(mo):
    mo.md(r"""2. **Test Results by Group**: Displays how test results (event $B$) are distributed across different groups""")
    return


@app.cell
def _(alt, calculations, mo, pd):
    # Test results by group (stacked bar)
    test_data = pd.DataFrame({
        'group': ['Group A', 'Group A', 'Group ¬A', 'Group ¬A'],
        'result_type': ['True Positive', 'False Negative', 'False Positive', 'True Negative'],
        'value': [calculations['true_positive'], calculations['false_negative'], 
                 calculations['false_positive'], calculations['true_negative']],
        'color': ['#FF6B6B', '#95E1D3', '#FFB347', '#4ECDC4']
    })

    test_chart = alt.Chart(test_data).mark_bar().encode(
        x=alt.X('group:N', title='Group'),
        y=alt.Y('value:Q', title='Probability'),
        color=alt.Color('result_type:N', scale=alt.Scale(range=['#FF6B6B', '#95E1D3', '#FFB347', '#4ECDC4'])),
        tooltip=['group', 'result_type', 'value']
    ).properties(
        title="Test Results by Group",
        width=400,
        height=300
    )

    mo.ui.altair_chart(test_chart)
    return


@app.cell
def _(mo):
    mo.md(r"""3. **Components Breakdown**: Shows key components in Bayes' theorem (prior, posterior, likelihood, marginal, etc.)""")
    return


@app.cell
def _(alt, calculations, mo, pd):
    # Bayes theorem components breakdown
    components_data = pd.DataFrame({
        'component': ["Prior P(A)", "Likelihood P(B|A)", "P(B|¬A)", "Marginal P(B)", "Posterior P(A|B)"],
        'value': [calculations['prior_a'], calculations['likelihood_b_given_a'], 
                 calculations['likelihood_b_given_not_a'], calculations['marginal_b'], 
                 calculations['posterior_a_given_b']],
        'color': ['#FF6B6B', '#4ECDC4', '#FFB347', '#95E1D3', '#9B59B6']
    })

    components_chart = alt.Chart(components_data).mark_bar().encode(
        x=alt.X('component:N', title='Component', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('value:Q', title='Probability'),
        color=alt.Color('color:N', scale=None),
        tooltip=['component', 'value']
    ).properties(
        title="Key Components of Bayes Theorem",
        width=500,
        height=300
    )

    mo.ui.altair_chart(components_chart)
    return


@app.cell
def _(mo):
    mo.md(r"""4. **Posterior Probability**: The final result - how much event $B$ changes our belief about event $A$""")
    return


@app.cell
def _(alt, calculations, mo, pd):
    # Prior vs Posterior comparison
    comparison_data = pd.DataFrame({
        'metric': ['Prior P(A)', 'Posterior P(A|B)'],
        'value': [calculations['prior_a'], calculations['posterior_a_given_b']],
        'color': ['#95E1D3', '#9B59B6']
    })

    comparison_chart = alt.Chart(comparison_data).mark_bar().encode(
        x=alt.X('metric:N', title='Probability Type'),
        y=alt.Y('value:Q', title='Value', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('color:N', scale=None),
        tooltip=['metric', 'value']
    ).properties(
        title="Prior vs Posterior Probability",
        width=300,
        height=300
    )

    mo.ui.altair_chart(comparison_chart)
    return


@app.cell
def _(mo):
    mo.md(r"""5. **Area Visualization**: A spatial representation where areas are proportional to probabilities""")
    return


@app.cell
def _(alt, calculations, mo, pd):
    # Create area-based visualization inspired by Oscar Bonilla's approach

    def create_venn_style_visualization(calc):
        # Create rectangles to represent different probability regions
        total_width = 10
        total_height = 10
        width_a = total_width * calc['prior_a']
        height_b_given_a = total_height * calc['likelihood_b_given_a']
        height_b_given_not_a = total_height * calc['likelihood_b_given_not_a']

        # Create data for rectangles
        rect_data = pd.DataFrame([
            {'x': 0, 'y': 0, 'x2': width_a, 'y2': total_height, 'region': 'A', 'opacity': 0.5, 'color': '#FF6B6B'},
            {'x': width_a, 'y': 0, 'x2': total_width, 'y2': total_height, 'region': '¬A', 'opacity': 0.5, 'color': '#4ECDC4'},
            {'x': 0, 'y': 0, 'x2': width_a, 'y2': height_b_given_a, 'region': 'A∩B', 'opacity': 0.8, 'color': '#FF6B6B'},
            {'x': width_a, 'y': 0, 'x2': total_width, 'y2': height_b_given_not_a, 'region': '¬A∩B', 'opacity': 0.8, 'color': '#4ECDC4'}
        ])

        # Create base chart
        selection = alt.selection_point()
        base = alt.Chart(rect_data).add_params(
            selection
        ).mark_rect().encode(
            x=alt.X('x:Q', scale=alt.Scale(domain=[-0.5, total_width+0.5])),
            y=alt.Y('y:Q', scale=alt.Scale(domain=[-0.5, total_height+0.5])),
            x2='x2:Q',
            y2='y2:Q',
            color=alt.Color('color:N', scale=None),
            opacity=alt.Opacity('opacity:Q'),
            stroke=alt.value('black'),
            strokeWidth=alt.value(1),
            tooltip=['region']
        )

        # Add text annotations with dynamic positioning
        text_data = pd.DataFrame([
            # Main region A - center of full A rectangle
            {'x': width_a/2, 'y': total_height/2, 'text': f"A\nP(A) = {calc['prior_a']:.3f}", 'fontSize': 14},
            # Main region ¬A - center of full ¬A rectangle  
            {'x': width_a + (total_width - width_a)/2, 'y': total_height/2, 'text': f"¬A\nP(¬A) = {calc['prior_not_a']:.3f}", 'fontSize': 14},
            # A∩B region - center of the B|A rectangle (only if it has reasonable size)
            {'x': width_a/2, 'y': height_b_given_a/2 if height_b_given_a > 1 else height_b_given_a + 0.5, 
             'text': f"A∩B\n{calc['true_positive']:.3f}", 'fontSize': 10 if height_b_given_a > 1 else 8},
            # ¬A∩B region - center of the B|¬A rectangle (only if it has reasonable size)
            {'x': width_a + (total_width - width_a)/2, 
             'y': height_b_given_not_a/2 if height_b_given_not_a > 1 else height_b_given_not_a + 0.5,
             'text': f"¬A∩B\n{calc['false_positive']:.3f}", 'fontSize': 10 if height_b_given_not_a > 1 else 8}
        ])

        text_chart = alt.Chart(text_data).mark_text(
            align='center',
            baseline='middle',
            color='black',
            fontWeight='bold'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N',
            size=alt.Size('fontSize:Q').scale(range=[8,16])
        )

        combined = (base + text_chart).resolve_scale(
            color='independent'
        ).properties(
            title="Area-Based Bayes Theorem Visualization",
            width=500,
            height=500
        )

        return combined

    venn_chart = create_venn_style_visualization(calculations)
    mo.ui.altair_chart(venn_chart)
    return


if __name__ == "__main__":
    app.run()
