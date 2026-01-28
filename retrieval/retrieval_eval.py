from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from deepeval.test_case import LLMTestCase
from retrieval_langchain import retrieve_chunks

from deepeval import evaluate


def generate_testcase(question):
    retrieved_responses = retrieve_chunks(question, k=5)
    test_case = LLMTestCase(
        input=question,
        actual_output=retrieved_responses[0].page_content,
        expected_output="""a. An Operator is aware of facts that indicate that the adverse incident was not related to
                        toxic effects or exposure from the pesticide application;
                        b. An Operator has been notified by EPA, and retains such notification, that the
                        reporting requirement has been waived for this incident or category of incidents;
                        c. An Operator receives information of an adverse incident, but that information is
                        clearly erroneous; or
                        d. An adverse incident occurs to pests that are similar in kind to potential target pests
                        identified on the FIFRA label, except as required in Part 6.4.3, Notification for
                        Adverse Incident to Threatened or Endangered Species or Critical Habit""",
        retrieval_context=[d.page_content for d in retrieved_responses]
    )

    return test_case

contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()

test1 = generate_testcase("When is adverse incident notification not required under PGP?")

print(evaluate(
    test_cases=[test1],
    metrics=[contextual_precision, contextual_recall, contextual_relevancy]
))
