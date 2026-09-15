"""CPU regression tests executing selected source methods without CUDA imports.

The real tensor operations and routing methods are used; CUDA graph capture and
full-model inference must additionally be validated on a GPU.
"""

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER = "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"


def method(path, cls, name):
    tree = ast.parse((ROOT / path).read_text())
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = {"torch": torch}
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), path, "exec"), ns)
    return ns[name], fn


class BackportTests(unittest.TestCase):
    def test_buffer_opt_in_and_defaults(self):
        create, _ = method(
            "python/sglang/srt/model_executor/runner_utils/buffers.py",
            "PrefillInputBuffers",
            "create",
        )
        args = dict(
            device="cpu",
            max_bs=2,
            max_num_tokens=8,
            cache_loc_dtype=torch.int64,
            hidden_size=4,
            dtype=torch.float32,
            enable_mamba_track=False,
        )
        self.assertIsNone(create(NS, is_multimodal=False, **args).input_embeds)
        self.assertEqual(
            create(NS, is_multimodal=True, **args).input_embeds.shape, (8, 4)
        )
        explicit = create(NS, is_multimodal=False, enable_input_embeds=True, **args)
        self.assertEqual(explicit.input_embeds.shape, (8, 4))
        self.assertIsNone(explicit.mrope_positions)

    def populate(self, mm):
        _, fn = method(RUNNER, "PrefillCudaGraphRunner", "load_batch")
        block = next(
            n
            for n in fn.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "self._use_draft_input_embeds"
        )
        embedding = torch.nn.Embedding(16, 4)
        ids = (
            torch.tensor([1, 1000000, 3, 4, 1000001, 6])
            if mm
            else torch.tensor([1, 2, 3, 4, 5, 6])
        )
        embeds = torch.randn(6, 4) if mm else None
        expected = embeds.clone() if mm else embedding(ids).detach().clone()
        if mm:
            with self.assertRaises(IndexError):
                embedding(ids)
            expected[torch.tensor([2, 5])] = embedding(torch.tensor([3, 6])).detach()
        buffer = torch.zeros(8, 4)
        ptr = buffer.data_ptr()
        owner = NS(
            _use_draft_input_embeds=True,
            layer_model=NS(embed_tokens=embedding),
            buffer_registry=NS(get_slot=lambda _: NS(buffer=buffer)),
        )
        batch = NS(
            mm_input_embeds=embeds,
            input_ids=ids,
            extend_start_loc=torch.tensor([0, 3]),
            extend_seq_lens=torch.tensor([3, 3]),
        )
        with torch.no_grad():
            exec(
                compile(
                    ast.fix_missing_locations(
                        ast.Module(body=[block], type_ignores=[])
                    ),
                    RUNNER,
                    "exec",
                ),
                {"self": owner, "forward_batch": batch, "num_tokens": 6},
            )
        torch.testing.assert_close(buffer[:6], expected)
        self.assertEqual(buffer.data_ptr(), ptr)
        self.assertTrue(torch.equal(buffer[6:], torch.zeros(2, 4)))

    def test_multimodal_ids_never_lookup_pad(self):
        self.populate(True)

    def test_text_embedding_path(self):
        self.populate(False)

    def test_replay_forwards_explicit_embeddings_and_restores_method(self):
        run, _ = method(RUNNER, "PrefillCudaGraphRunner", "_execute_body_capture")
        original = lambda: None
        layer = NS(forward=original)
        embeds = torch.randn(4, 4)
        static = NS(
            input_ids=torch.tensor([1, 1000000, 2, 0]),
            positions=torch.arange(4),
            input_embeds=embeds,
        )

        def outer(ids, pos, batch, **kw):
            self.assertIs(kw["input_embeds"], embeds)
            self.assertEqual(kw["sentinel"], 7)
            return layer.forward()

        expected = torch.ones(4, 4)
        owner = NS(
            _is_full_backend=False,
            _input_embeds_arg_idx=None,
            _use_draft_input_embeds=True,
            layer_model=layer,
            model_runner=NS(model=NS(forward=outer)),
            buffer_registry=NS(has_slot=lambda _: False),
            backend=NS(replay=lambda *a, **kw: expected),
            _prefill_forward_context=lambda *a, **kw: contextlib.nullcontext(),
        )
        self.assertIs(run(owner, static, static, 4, 3, None, sentinel=7), expected)
        self.assertIs(layer.forward, original)
        owner.model_runner.model.forward = lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("test failure")
        )
        with self.assertRaises(ValueError):
            run(owner, static, static, 4, 3, None)
        self.assertIs(layer.forward, original)

    def test_layer_capture_delegate_and_legacy_fallback(self):
        inner, _ = method(
            "python/sglang/srt/models/qwen3_5.py",
            "Qwen3_5ForCausalLM",
            "set_eagle3_layers_to_capture",
        )
        outer, _ = method(
            "python/sglang/srt/models/qwen3_vl.py",
            "Qwen3VLForConditionalGeneration",
            "set_eagle3_layers_to_capture",
        )
        backbone = NS(layers=[NS() for _ in range(12)])
        backbone.set_eagle3_layers_to_capture = lambda ids: inner(backbone, ids)
        wrapper = NS(
            pp_group=NS(is_last_rank=True),
            model=backbone,
            config=NS(num_hidden_layers=12),
        )
        outer(wrapper, [1, 4, 8])
        self.assertEqual(backbone.layers_to_capture, [2, 5, 9])
        self.assertEqual(
            [
                i
                for i, x in enumerate(backbone.layers)
                if getattr(x, "_is_layer_to_capture", False)
            ],
            [2, 5, 9],
        )
        wrapper.model = NS()
        outer(wrapper, [1, 4, 8])
        self.assertEqual(wrapper.model.layers_to_capture, [2, 5, 9])
        wrapper.pp_group.is_last_rank = False
        wrapper.model = NS()
        outer(wrapper, [1, 4, 8])
        self.assertFalse(hasattr(wrapper.model, "layers_to_capture"))


if __name__ == "__main__":
    unittest.main()
