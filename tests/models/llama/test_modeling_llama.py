# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the PyTorch LLaMA model. """

import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import LlamaConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        CodeLlamaTokenizer,
        LlamaForCausalLM,
        LlamaForSequenceClassification,
        LlamaModel,
        LlamaTokenizer,
    )


class LlamaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LlamaModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = LlamaModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = LlamaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = LlamaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class LlamaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (LlamaModel, LlamaForCausalLM, LlamaForSequenceClassification) if is_torch_available() else ()
    all_generative_model_classes = (LlamaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": LlamaModel,
            "text-classification": LlamaForSequenceClassification,
            "text-generation": LlamaForCausalLM,
            "zero-shot": LlamaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = LlamaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlamaConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_llama_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = LlamaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_llama_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = LlamaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_llama_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = LlamaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    @unittest.skip("Llama buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = LlamaModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = LlamaModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            self.assertTrue(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    @require_flash_attn
    @require_torch_gpu
    @require_bitsandbytes
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_generate_padding_right(self):
        """
        Overwritting the common test as the test is flaky on tiny models
        """
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            load_in_4bit=True,
            device_map={"": 0},
        )

        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        texts = ["hi", "Hello this is a very long sentence"]

        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(0)

        output_native = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_native = tokenizer.batch_decode(output_native)

        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", load_in_4bit=True, device_map={"": 0}, attn_implementation="flash_attention_2"
        )

        output_fa_2 = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_fa_2 = tokenizer.batch_decode(output_fa_2)

        self.assertListEqual(output_native, output_fa_2)

    @require_flash_attn
    @require_torch_gpu
    @slow
    def test_use_flash_attention_2_true(self):
        """
        NOTE: this is the only test testing that the legacy `use_flash_attention=2` argument still works as intended.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(config)
                model.save_pretrained(tmp_dir)

                new_model = LlamaForCausalLM.from_pretrained(
                    tmp_dir, use_flash_attention_2=True, torch_dtype=torch.float16
                ).to("cuda")

                self.assertTrue(new_model.config._attn_implementation == "flash_attention_2")

                has_flash = False
                for name, submodule in new_model.named_modules():
                    if "FlashAttention" in submodule.__class__.__name__:
                        has_flash = True
                        break
                if not has_flash:
                    raise ValueError("The flash model should have flash attention layers")

    @require_torch_sdpa
    @slow
    def test_eager_matches_sdpa_generate(self):
        """
        Overwritting the common test as the test is flaky on tiny models
        """
        max_new_tokens = 30

        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        model_sdpa = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(torch_device)

        self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

        model_eager = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(torch_device)

        self.assertTrue(model_eager.config._attn_implementation == "eager")

        for name, submodule in model_eager.named_modules():
            if "SdpaAttention" in submodule.__class__.__name__:
                raise ValueError("The eager model should not have SDPA attention layers")

        has_sdpa = False
        for name, submodule in model_sdpa.named_modules():
            if "SdpaAttention" in submodule.__class__.__name__:
                has_sdpa = True
                break
        if not has_sdpa:
            raise ValueError("The SDPA model should have SDPA attention layers")

        texts = ["hi", "Hello this is a very long sentence my friend", "Today I am in Paris and"]

        for padding_side in ["left", "right"]:
            tokenizer.padding_side = padding_side
            tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(torch_device)

            res_eager = model_eager.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            res_sdpa = model_sdpa.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            self.assertTrue(torch.allclose(res_eager, res_sdpa))


@require_torch
class LlamaIntegrationTest(unittest.TestCase):
    @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    @require_read_token
    def test_llama_3_1_hard(self):
        """
        An integration test for llama 3.1. It tests against a long output to ensure the subtle numerical differences
        from llama 3.1.'s RoPE can be detected
        """
        # diff on `EXPECTED_TEXT`:
        # 2024-08-26: updating from torch 2.3.1 to 2.4.0 slightly changes the results.
        EXPECTED_TEXT = (
            "Tell me about the french revolution. The french revolution was a period of radical political and social "
            "upheaval in France that lasted from 1789 until 1799. It was a time of great change and upheaval, marked "
            "by the overthrow of the monarchy, the rise of the middle class, and the eventual establishment of the "
            "First French Republic.\nThe revolution began in 1789 with the Estates-General, a representative "
            "assembly that had not met since 1614. The Third Estate, which represented the common people, "
            "demanded greater representation and eventually broke away to form the National Assembly. This marked "
            "the beginning of the end of the absolute monarchy and the rise of the middle class.\n"
        )

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16
        )
        input_text = ["Tell me about the french revolution."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    @slow
    @require_read_token
    def test_model_7b_logits_bf16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        EXPECTED_MEAN = {
            7: torch.tensor([[-6.5061, -4.1147, -4.9669, -3.2038, 0.8069, -2.9694, 1.2864, -3.3786]]),
            8: torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]])
        }

        self.assertTrue(torch.allclose(EXPECTED_MEAN[self.cuda_compute_capability_major_version].to(torch_device), out.logits.mean(-1), atol=1e-2, rtol=1e-2))

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = {
            7: torch.tensor([[-12.5000, -7.0625, -0.6289, -7.8750, -6.9688, -7.8125, -6.4688, -7.4375, -7.6875, -6.9375, -6.0312, -7.0000, -1.8594, 1.8438, -8.5000]]),
            8: torch.tensor([[-12.5625,  -7.1250,  -0.6289,  -7.8750,  -6.9688,  -7.8125,  -6.5000, -7.4375,  -7.6562,  -6.9688,  -6.0312,  -7.0312,  -1.8203,   1.8750, -8.5000]])
        }
        # fmt: on

        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits[0, 0, :15],
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @slow
    @require_read_token
    def test_model_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
        out = model(torch.tensor([input_ids]))
        # Expected mean on dim = -1
        EXPECTED_MEAN = {
            7: torch.tensor([[-6.6420, -4.1227, -4.9809, -3.2041, 0.8261, -3.0052, 1.2957, -3.3648]]),
            8: torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]])
        }

        self.assertTrue(torch.allclose(EXPECTED_MEAN[self.cuda_compute_capability_major_version].to(torch_device), out.logits.mean(-1), atol=1e-2, rtol=1e-2))

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = {
            7: torch.tensor([-12.8125, -7.3359, -0.4846, -8.0234, -7.2383, -7.9922, -6.4805, -7.7344, -7.8125, -7.0078, -6.1797, -7.1094, -1.8633, 1.9736, -8.6016]),
            8: torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328])
        }
        # fmt: on

        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits[0, 0, :15],
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    def test_model_13b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto")
        out = model(torch.tensor(input_ids))
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-2.0622, -1.2794, -1.1638, -0.9788, -1.4603, -1.0238, -1.7893, -1.4411]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-8.1406, -8.0547, 2.7461, -1.2344, -0.1448, -1.8262, -1.0020, -1.8154, -1.6895, -1.8516, -2.3574, -0.9277, 3.7598, 6.5742, -1.2998, -0.1177, -8.1406, -2.9688, -2.9199, -3.1699, -3.5254, -2.3555, -2.7988, -3.4141, -2.8262, -4.5195, -3.3379, -3.3164, -2.7832, -3.0273])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    def test_model_13bf_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map="auto")
        out = model(torch.tensor(input_ids))
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-0.8562, -1.8520, -0.7551, -0.4162, -1.5161, -1.2038, -2.4823, -2.3254]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-2.2227, 4.8828, 0.9023, -0.4578, -0.7871, -0.1033, -0.6221, -0.5786, -0.7803, -1.0674, -1.2920, -0.1570, 0.8008, 2.0723, -0.9497, 0.2771, -2.2227, -0.7612, -1.4346, -1.2061, -1.6426, -0.3000, -0.7139, -1.1934, -1.8691, -1.6973, -1.5947, -1.2705, -0.3523, -0.5513])  # fmt: skip
        torch.testing.assert_close(out.mean(-1), EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    @unittest.skip(
        "Logits are not exactly the same, once we fix the instabalities somehow, will update! Also it is gonna be a `too_slow` test"
    )
    @slow
    def test_model_70b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", device_map="auto")
        out = model(torch.tensor(input_ids))

        EXPECTED_MEAN = torch.tensor(
            [[-4.2327, -3.3360, -4.6665, -4.7631, -1.8180, -3.4170, -1.4211, -3.1810]], dtype=torch.float32
        )
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        EXPECTED_SLICE = torch.tensor([-9.4922, -3.9551, 1.7998, -5.6758, -5.1055, -5.8984, -4.8320, -6.8086, -6.5391, -5.6172, -5.5820, -5.5352, 1.7881, 3.6289, -6.5117, -3.4785, -9.5000, -6.0352, -6.8125, -6.0195, -6.6836, -5.4727, -6.2812, -6.0391, -7.3398, -7.4297, -7.4844, -6.5820, -5.8789, -5.5312])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @unittest.skip("Model is curently gated")
    @slow
    def test_model_13b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same everywhere in the universe and 2) the passage of time and the length of objects can vary depending on the observer\'s frame of reference.\n\nThe first part of the theory, that the laws of physics are the same everywhere, is known as the "princi"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-chat-hf", device_map="sequential", use_safetensors=False
        )

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)


@require_torch
class CodeLlamaIntegrationTest(unittest.TestCase):
    PROMPTS = [
        '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
''',
        """# Installation instructions:
    ```bash
<FILL_ME>
    ```
This downloads the LLaMA inference code and installs the repository as a local pip package.
""",
        """class InterfaceManagerFactory(AbstractManagerFactory):
    def __init__(<FILL_ME>
def main():
    factory = InterfaceManagerFactory(start=datetime.now())
    managers = []
    for i in range(10):
        managers.append(factory.build(id=i))
""",
        """/-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/
theorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :
π₁ P = 0 ↔ <FILL_ME> = 0 :=
begin
split,
{ intros h f,
    rw pi_1_etalisation at h,
    simp [h],
    refl
},
{ intro h,
    have := @quasi_adjoint C D P,
    simp [←pi_1_etalisation, this, h],
    refl
}
end
""",
    ]

    @require_torch_accelerator
    @slow
    def test_model_7b_logits(self):
        model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf").to(torch_device)
        tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        # Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
        # meaning by default this supports passing splitted list of inputs
        processed_text = tokenizer.batch_decode(tokenizer(self.PROMPTS)["input_ids"], add_special_tokens=False)
        # fmt: off
        EXPECTED_TEXT = [
            '<s> <PRE> def remove_non_ascii(s: str) -> str:\n    """  <SUF>\n    return result\n <MID>',
            '<s> <PRE> # Installation instructions:\n    ```bash\n <SUF>\n    ```\nThis downloads the LLaMA inference code and installs the repository as a local pip package.\n <MID>',
            '<s> <PRE> class InterfaceManagerFactory(AbstractManagerFactory):\n    def __init__( <SUF>\ndef main():\n    factory = InterfaceManagerFactory(start=datetime.now())\n    managers = []\n    for i in range(10):\n        managers.append(factory.build(id=i))\n <MID>',
            '<s> <PRE> /-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/\ntheorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :\nπ₁ P = 0 ↔  <SUF> = 0 :=\nbegin\nsplit,\n{ intros h f,\n    rw pi_1_etalisation at h,\n    simp [h],\n    refl\n},\n{ intro h,\n    have := @quasi_adjoint C D P,\n    simp [←pi_1_etalisation, this, h],\n    refl\n}\nend\n <MID>'
        ]
        # fmt: on
        self.assertEqual(processed_text, EXPECTED_TEXT)
        processed_text_suffix_first = tokenizer.batch_decode(
            tokenizer(self.PROMPTS, suffix_first=True, add_special_tokens=False)["input_ids"]
        )

        # fmt: off
        EXPECTED_TEXT = [
            '<PRE> <SUF>\n    return result\n <MID> def remove_non_ascii(s: str) -> str:\n    """ ',
            '<PRE> <SUF>\n    ```\nThis downloads the LLaMA inference code and installs the repository as a local pip package.\n <MID> # Installation instructions:\n    ```bash\n',
            '<PRE> <SUF>\ndef main():\n    factory = InterfaceManagerFactory(start=datetime.now())\n    managers = []\n    for i in range(10):\n        managers.append(factory.build(id=i))\n <MID> class InterfaceManagerFactory(AbstractManagerFactory):\n    def __init__(',
            '<PRE> <SUF> = 0 :=\nbegin\nsplit,\n{ intros h f,\n    rw pi_1_etalisation at h,\n    simp [h],\n    refl\n},\n{ intro h,\n    have := @quasi_adjoint C D P,\n    simp [←pi_1_etalisation, this, h],\n    refl\n}\nend\n <MID> /-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/\ntheorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :\nπ₁ P = 0 ↔ '
        ]
        EXPECTED_IDS = torch.tensor([[    1, 32007, 822, 3349, 29918, 5464, 29918, 294, 18869, 29898,29879, 29901, 851, 29897, 1599, 851, 29901, 13, 1678, 9995, 29871, 32008, 13, 1678, 736, 1121, 13, 32009, 15941, 1661, 29899, 28599, 2687, 4890, 515, 263, 1347, 29889, 13, 13, 1678, 826, 3174, 29901, 13, 4706, 269, 29901, 450, 1347, 304, 3349, 1661, 29899, 28599, 2687, 4890, 515, 29889, 13, 13, 1678, 16969, 29901, 13, 4706, 450, 1347, 411, 1661, 29899, 28599, 2687, 4890, 6206, 29889, 13, 1678, 9995, 13, 1678, 1121, 353, 5124, 13, 1678, 363, 274, 297, 269, 29901, 13, 4706, 565, 4356, 29898, 29883, 29897, 529, 29871, 29896, 29906, 29947, 29901, 13, 9651, 1121, 4619, 274, 32010, 2]])
        # fmt: on
        self.assertEqual(processed_text_suffix_first, EXPECTED_TEXT)
        input_ids = tokenizer(self.PROMPTS[0], return_tensors="pt")["input_ids"]
        generated_ids = model.generate(input_ids.to(torch_device), max_new_tokens=128)
        torch.testing.assert_close(generated_ids, EXPECTED_IDS)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        model._cache = None  # clear cache object, initialized when we pass `cache_implementation="static"`
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)


@slow
@require_torch_gpu
class Mask4DTestHard(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def setUp(self):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model_dtype = torch.float32
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=self.model_dtype).to(torch_device)

    def get_test_data(self):
        template = "my favorite {}"
        items = ("pet is a", "artist plays a", "name is L")  # same number of tokens in each item

        batch_separate = [template.format(x) for x in items]  # 3 separate lines
        batch_shared_prefix = template.format(" ".join(items))  # 1 line with options concatenated

        input_ids = self.tokenizer(batch_separate, return_tensors="pt").input_ids.to(torch_device)
        input_ids_shared_prefix = self.tokenizer(batch_shared_prefix, return_tensors="pt").input_ids.to(torch_device)

        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                ]
            ],
            device=torch_device,
        )

        position_ids = torch.arange(input_ids.shape[1]).tile(input_ids.shape[0], 1).to(torch_device)

        # building custom positions ids based on custom mask
        position_ids_shared_prefix = (mask_shared_prefix.sum(dim=-1) - 1).reshape(1, -1)
        # effectively: position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]]).to(device)

        # inverting the mask
        min_dtype = torch.finfo(self.model_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=self.model_dtype) * min_dtype

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix

    def test_stacked_causal_mask(self):
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # single forward run with 4D custom mask
        logits_shared_prefix = self.model.forward(
            input_ids_shared_prefix, attention_mask=mask_shared_prefix, position_ids=position_ids_shared_prefix
        ).logits
        logits_shared_prefix_last = logits_shared_prefix[
            0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1], :
        ]  # last three tokens
        decoded_shared_prefix = [self.tokenizer.decode(t) for t in logits_shared_prefix_last.argmax(dim=-1)]

        self.assertEqual(decoded, decoded_shared_prefix)

    def test_partial_stacked_causal_mask(self):
        # Same as the test above, but the input is passed in two groups. It tests that we can pass partial 4D attention masks

        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # 2 forward runs with custom 4D masks
        part_a = 3  # split point

        input_1a = input_ids_shared_prefix[:, :part_a]
        position_ids_1a = position_ids_shared_prefix[:, :part_a]
        mask_1a = mask_shared_prefix[:, :, :part_a, :part_a]

        outs_1a = self.model.forward(input_1a, attention_mask=mask_1a, position_ids=position_ids_1a)
        past_key_values_a = outs_1a["past_key_values"]

        # Case 1: we pass a 4D attention mask regarding the current sequence length (i.e. [..., seq_len, full_len])
        input_1b = input_ids_shared_prefix[:, part_a:]
        position_ids_1b = position_ids_shared_prefix[:, part_a:]
        mask_1b = mask_shared_prefix[:, :, part_a:, :]
        outs_1b = self.model.forward(
            input_1b,
            attention_mask=mask_1b,
            position_ids=position_ids_1b,
            past_key_values=past_key_values_a,
        )
        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[
                0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1] - part_a
            ]
        ]
        infilling = tokenizer.batch_decode(generated_ids)
        self.assertEqual(infilling, EXPECTED_INFILLING)
