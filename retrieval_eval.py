from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)

from deepeval.test_case import LLMTestCase
from retrieval.retrieval_langchain import retrieve_chunks

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig

from api import _answer_with_llm, _build_excerpt_block

import os
import json

os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = '0'


def generate_testcase(question, expected_output):
    retrieved_responses = retrieve_chunks(question, k=5)
    excerpts, _ = _build_excerpt_block(retrieved_responses)
    test_case = LLMTestCase(
        input=question,
        actual_output=_answer_with_llm(question, excerpts),
        #expected_output=expected_output,
        retrieval_context=[d.page_content for d in retrieved_responses]
    )

    return test_case


def run():
    # How relevant the LLM answer is to the question - uses LLM as judge
    answer_relevancy = AnswerRelevancyMetric(threshold=0.6)

    # Is the response grounded in evidence from retrieved passages - no halluncations?
    faithfulness = FaithfulnessMetric(threshold=0.8)

    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    # contextual_relevancy = ContextualRelevancyMetric()

    test1 = generate_testcase("When is adverse incident notification not required under PGP?",
                                """a. An Operator is aware of facts that indicate that the adverse incident was not related to
                            toxic effects or exposure from the pesticide application;
                            b. An Operator has been notified by EPA, and retains such notification, that the
                            reporting requirement has been waived for this incident or category of incidents;
                            c. An Operator receives information of an adverse incident, but that information is
                            clearly erroneous; or
                            d. An adverse incident occurs to pests that are similar in kind to potential target pests
                            identified on the FIFRA label, except as required in Part 6.4.3, Notification for
                            Adverse Incident to Threatened or Endangered Species or Critical Habit""")

    # tests = [test1]

    questions_file = "pgp_test_questions.json"

    answer_relevancy_mean = 0
    faithfulness_mean = 0
    count = 0

    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            for i, question in enumerate(questions):
                if i == 1 or i >= 5:
                    test = generate_testcase(question["question"], None)
                    
                    print("Input: ", test.input)
                    print("Answer: ", test.actual_output)

                    answer_relevancy.measure(test)
                    score = answer_relevancy.score

                    answer_relevancy_mean += score
                    print("AR Score: ", score)
                    print("AR Reason: ", answer_relevancy.reason)

                    faithfulness.measure(test)
                    faith_score = faithfulness.score
                    faithfulness_mean += faith_score

                    print("Faithfulness Score: ", faith_score)
                    print("Faithfulness Reason: ", faithfulness.reason)

                    count += 1
    except Exception as e:
        print(e)
    
    print("AR Mean: ", answer_relevancy_mean / count)
    print("Faithfulness Mean: ", faithfulness_mean / count)


    # print(tests)

    # async_config = AsyncConfig(max_concurrent=2, throttle_value=5) # Throttle - how many seconds to terminate the test case

    # print(evaluate(
    #     test_cases=tests,
    #     metrics=[answer_relevancy, faithfulness, 
    #             #contextual_precision, contextual_recall
    #             ],
    #     async_config=async_config
    # ))

if __name__ == "__main__":
    run()
    
# contextual_precision.measure(test1)
# print("Score: ", contextual_precision.score)
# print("Reason: ", contextual_precision.reason)

# contextual_recall.measure(test1)
# print("Score: ", contextual_recall.score)
# print("Reason: ", contextual_recall.reason)

# contextual_relevancy.measure(test1)
# print("Score: ", contextual_relevancy.score)
# print("Reason: ", contextual_relevancy.reason)
