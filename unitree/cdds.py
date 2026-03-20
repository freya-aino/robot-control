import os
import time
from cyclonedds import builtin, domain
from tqdm import tqdm

def get_all_cdds_topics(
    domain_id: int = 0,
    timeout_s: float = 5
):
    # os.environ["CYCLONEDDS_NETWORK_INTERFACE"] = NETWORK_INTERFACE.LAPTOP_1.value
    local_file_path = os.path.dirname(os.path.abspath(__file__))
    os.environ["CYCLONEDDS_URI"] = os.path.join(local_file_path, "cyclonedds.xml")
    # TODO, CHANGE THIS TO THE CORRECT GLOBAL RELATIVE PATH

    dp = domain.DomainParticipant(
        domain_id=domain_id
    )
    reader = builtin.BuiltinDataReader(
        subscriber_or_participant=dp,
        builtin_topic=builtin.BuiltinTopicDcpsTopic
    )

    all_topics = {}
    deadline = time.monotonic() + timeout_s
    with tqdm(
        desc="scanning",
        total=timeout_s,
        # leave=False,
        bar_format="{desc} ({postfix}): {bar}{remaining}",
    ) as prog_bar:
        while time.monotonic() < deadline:
            samples = reader.take()
            for sample in samples:
                all_topics[sample.topic_name] = sample.type_name
            prog_bar.n = timeout_s - (deadline - time.monotonic())
            prog_bar.refresh()
            prog_bar.set_postfix(topics_found=len(all_topics))
    return all_topics
