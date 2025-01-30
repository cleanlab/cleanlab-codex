# """
# This module provides validation functions for checking if an LLM response is unhelpful.
# """

# from typing import Optional, TYPE_CHECKING

# if TYPE_CHECKING:
#     try:
#         from cleanlab_studio.studio.trustworthy_language_model import TLM  # noqa: F401
#     except ImportError:
#         raise ImportError("The 'cleanlab_studio' library is required to run this validator. Please install it with `pip install cleanlab-studio`.")


# def is_bad_response(response: str, fallback_answer: str, threshold: int = 70) -> bool:
#     """Use partial ratio to match a fallback_answer to the response, indicating how unhelpful the response is.
#     If the partial ratio is greater than or equal to the threshold, return True.
#     """
#     try:
#         from thefuzz import fuzz
#     except ImportError:
#         raise ImportError("The 'thefuzz' library is required to run this validator. Please install it with `pip install thefuzz`.")

#     partial_ratio = fuzz.partial_ratio(fallback_answer.lower(), response.lower())
#     return partial_ratio >= threshold

# def is_bad_response_contains_phrase(response: str, fallback_responses: list[str]) -> bool:
#     """Check whether the response matches a fallback phrase, indicating the response is not helpful.

#     Args:
#         response: The response from the assistant
#         fallback_responses: A list of fallback phrases to check against the response.

#     Returns:
#         bool: True if the response appears to be a fallback/inadequate response
#     """
#     return any(
#         phrase.lower() in response.lower()
#         for phrase in fallback_responses
#     )

# def is_bad_response_untrustworthy(response: str, context: str, query: str, tlm: TLM, threshold: float = 0.5) -> bool:
#     """
#     Check whether the response is untrustworthy based on the TLM score.
    
#     Args:
#         response: The response from the assistant
#         context: The context of the query
#         query: The user query
#         tlm: The TLM model to use
#         threshold: The threshold for the TLM score. If the score is less than this threshold, the response is considered untrustworthy.

#     Returns:
#         bool: True if the response is untrustworthy
#     """
#     prompt = f"Context: {context}\n\n Query: {query}\n\n Query: {query}"
#     resp = tlm.get_trustworthiness_score(prompt, response)
#     score = resp['trustworthiness_score']
#     return score < threshold

# # TLM Binary Classification
# def is_bad_response_unhelpful(response: str, tlm: TLM, query: Optional[str] = None, trustworthiness_score_threshold: Optional[float] = None) -> bool:
#     """
#     Check whether the response is unhelpful based on the TLM score. A query may optionally be provided to help the TLM determine if the response is helpful to answer the given query.
    
#     Args:
#         response: The response from the assistant
#         tlm: The TLM model to use
#         query: The user query
#         trustworthiness_score_threshold: The threshold for the TLM score. If the score is less than this threshold, the response is considered unhelpful.

#     Returns:
#         bool: True if the response is unhelpful
#     """
#     if query is None:
#         prompt = (
#             "Consider the following AI Assistant Response.\n\n"
#             f"AI Assistant Response: {response}\n\n"
#             "Is the AI Assistant Response helpful? Answer Yes/No only."
#         )
#     else:
#         prompt = (
#             "Consider the following User Query and AI Assistant Response.\n\n"
#             f"User Query: {query}\n\n"
#             f"AI Assistant Response: {response}\n\n"
#             "Is the AI Assistant Response helpful? Answer Yes/No only."
#         )
#     output = tlm.prompt(prompt, constrain_outputs=["Yes", "No"])
#     response_marked_unhelpful = output["response"].lower() == "no"
#     # TODO: Decide if we should keep the trustworthiness score threshold.
#     is_trustworthy = trustworthiness_score_threshold is None or (output["trustworthiness_score"] > trustworthiness_score_threshold)
#     return response_marked_unhelpful and is_trustworthy
