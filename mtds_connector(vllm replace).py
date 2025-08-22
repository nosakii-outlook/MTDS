# SPDX-License-Identifier: Apache-2.0
"""
MTDS KV Cache Connector for Distributed Machine Learning Inference

The MTDSConnector can (1) transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using MTDS;
(2) offload and share KV caches.
"""

# Standard
from typing import TYPE_CHECKING, List, Tuple, Union

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
import torch

if TYPE_CHECKING:
    # Third Party
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class MTDSConnector(KVConnectorBase):
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config

        # First Party
        from mtds.integration.vllm.utils import ENGINE_NAME
        from mtds.integration.vllm.vllm_adapter import (
            RetrieveStatus,
            StoreStatus,
            init_mtds_engine,
            mtds_retrieve_kv,
            mtds_should_retrieve,
            mtds_should_store,
            mtds_store_kv,
        )
        from mtds.v1.cache_engine import MTDSEngineBuilder

        logger.info(
            "Initializing MTDSConfig under kv_transfer_config %s",
            self.transfer_config,
        )

        # TODO (Jiayi): Find model_config, parallel_config, and cache_config
        self.engine = init_mtds_engine(
            config.model_config,
            config.parallel_config,
            config.cache_config,
            config.scheduler_config,
        )
        self.mtds_engine_name = ENGINE_NAME
        self.mtds_engine_builder = MTDSEngineBuilder

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
    self.mtds_retrieve_kv = mtds_retrieve_kv
    self.mtds_store_kv = mtds_store_kv
    self.mtds_should_store = mtds_should_store
    self.mtds_should_retrieve = mtds_should_retrieve
    self.store_status = StoreStatus
    self.retrieve_status = RetrieveStatus

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
    ) -> Tuple[
        Union[torch.Tensor, IntermediateTensors],
        bool,
        "ModelInputForGPUWithSamplingMetadata",
    ]:
        hidden_or_intermediate_states = None

        # TODO (Jiayi): Need to support chunked prefill
        retrieve_status = self.mtds_should_retrieve(model_input)

        model_input, bypass_model_exec, hidden_or_intermediate_states = (
            self.mtds_retrieve_kv(
                model_executable,
                model_input,
                self.cache_config,
                kv_caches,
                retrieve_status,
            )
        )

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
    ) -> None:
        # TODO (Jiayi): Only normal prefill is supported for now
        store_status = self.mtds_should_store(model_input)
        self.mtds_store_kv(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            model_executable,
            model_input,
            kv_caches,
            store_status,
        )

    def close(self):
    self.mtds_engine_builder.destroy(self.mtds_engine_name)
