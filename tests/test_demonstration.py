from copy import deepcopy

import torch

from wan_va.modules.demonstration import DemonstrationAttention, DemonstrationEncoder
from wan_va.modules.model import WanTransformer3DModel


def _encoder() -> DemonstrationEncoder:
    return DemonstrationEncoder(
        in_channels=2,
        patch_size=(1, 2, 2),
        attention_dim=8,
    )


def _model(**kwargs) -> WanTransformer3DModel:
    settings = {
        "patch_size": [1, 2, 2],
        "num_attention_heads": 2, "attention_head_dim": 12,
        "in_channels": 2, "out_channels": 2, "action_dim": 3,
        "text_dim": 6, "freq_dim": 4, "ffn_dim": 16,
        "num_layers": 2,
        "rope_max_seq_len": 16, "attn_mode": "torch",
        "mcp_hidden_collect_layers": (0,),
    }
    settings.update(kwargs)
    return WanTransformer3DModel(**settings)


def _train_input(
    demo_latents: torch.Tensor, *, include_mcp: bool = False
) -> dict:
    batch_size = demo_latents.shape[0]
    video = torch.randn(batch_size, 2, 1, 2, 2)
    action = torch.randn(batch_size, 3, 1, 1, 1)
    video_grid = torch.zeros(batch_size, 4, 1, dtype=torch.long)
    action_grid = torch.zeros(batch_size, 4, 1, dtype=torch.long)
    payload = {
        "latent_dict": {
            "noisy_latents": video.clone(),
            "latent": video.clone(),
            "text_emb": torch.randn(batch_size, 2, 6),
            "grid_id": video_grid,
            "timesteps": torch.ones(batch_size, 1),
            "cond_timesteps": torch.zeros(batch_size, 1),
        },
        "action_dict": {
            "noisy_latents": action.clone(),
            "latent": action.clone(),
            "grid_id": action_grid,
            "timesteps": torch.ones(batch_size, 1),
            "cond_timesteps": torch.zeros(batch_size, 1),
        },
        "demo_latents": demo_latents,
        "demo_positions": torch.tensor([[0.0, 1.0]]).expand(batch_size, -1),
        "demo_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "chunk_size": 1,
        "window_size": 1,
    }
    if include_mcp:
        payload["mcp_latent_dicts"] = [{"noisy_latents": video.clone(),
                                        "grid_id": video_grid,
                                        "timesteps": torch.ones(batch_size, 1)}]
    return payload


def test_encoder_masks_padded_frames_and_preserves_token_order():
    # Given distinct temporal/spatial patches and one padded frame.
    encoder = _encoder()
    with torch.no_grad():
        encoder.patch_projection.weight.copy_(torch.eye(8))
        encoder.patch_projection.bias.zero_()
        for parameter in encoder.position_projection.parameters():
            parameter.zero_()
    latents = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 2, 4)
    positions = torch.tensor([[0.0, 0.5, 1.0]])
    frame_mask = torch.tensor([[True, True, False]])

    # When the demonstration is patch encoded.
    tokens, token_mask = encoder(latents, positions, frame_mask)

    # Then temporal-major patch order and the padded-frame mask are retained.
    expected = torch.tensor([[
        [0, 1, 4, 5, 24, 25, 28, 29],
        [2, 3, 6, 7, 26, 27, 30, 31],
        [8, 9, 12, 13, 32, 33, 36, 37],
        [10, 11, 14, 15, 34, 35, 38, 39],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]], dtype=torch.float32)
    assert torch.equal(tokens, expected)
    assert token_mask.tolist() == [[True, True, True, True, False, False]]


def test_attention_is_exact_zero_at_initialization_and_all_masked_after_training():
    # Given a zero-output initialized demonstration attention branch.
    attention = DemonstrationAttention(model_dim=8, attention_dim=8, num_heads=2)
    query = torch.randn(1, 3, 8)
    demo = torch.randn(1, 2, 8)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    # When a valid demonstration is attended before training.
    initial = attention(query, demo, valid_mask)

    # Then the residual contribution is bitwise zero.
    assert torch.equal(initial, torch.zeros_like(initial))

    # Given a trained output projection with nonzero bias.
    with torch.no_grad():
        attention.output.weight.fill_(0.1)
        attention.output.bias.fill_(0.2)

    # When every demonstration token is masked.
    masked = attention(query, demo, torch.zeros_like(valid_mask))

    # Then the branch still bypasses exactly, including its output bias.
    assert torch.equal(masked, torch.zeros_like(masked))


def test_attention_isolates_packed_samples_and_ignores_padding():
    # Given packed query tokens from two samples and distinct demonstrations.
    attention = DemonstrationAttention(model_dim=8, attention_dim=8, num_heads=2)
    with torch.no_grad():
        attention.output.weight.copy_(torch.eye(8))
    query = torch.randn(1, 4, 8)
    demo = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    sample_ids = torch.tensor([0, 0, 1, 1])

    # When only the other sample and masked padding are changed.
    baseline = attention(query, demo, mask, query_sample_ids=sample_ids)
    changed = demo.clone()
    changed[1] += 100
    changed[0, 2] -= 100
    result = attention(query, changed, mask, query_sample_ids=sample_ids)

    # Then sample zero is isolated and masked padding has no effect.
    assert torch.allclose(baseline[:, :2], result[:, :2])
    assert not torch.allclose(baseline[:, 2:], result[:, 2:])


def test_cached_single_demo_broadcasts_and_survives_prediction_refresh():
    # Given a prepared one-sample cache.
    attention = DemonstrationAttention(model_dim=8, attention_dim=8, num_heads=2)
    with torch.no_grad():
        attention.output.weight.copy_(torch.eye(8))
    demo = torch.randn(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    sentinel = attention(None, demo, mask, cache_name="episode", prepare_cache=True)

    # When a two-branch CFG query reuses it and prediction caches are cleared.
    attention.clear_pred_cache("episode")
    query = torch.randn(2, 3, 8)
    cached = attention(query, cache_name="episode")
    direct = attention(query, demo, mask)

    # Then cache preparation returns None and both branches match direct attention.
    assert sentinel is None
    assert torch.allclose(cached, direct)

    # When the full cache is cleared, reuse fails explicitly.
    attention.clear_cache("episode")
    try:
        attention(query, cache_name="episode")
    except RuntimeError as error:
        assert "demonstration cache" in str(error)
    else:
        raise AssertionError("missing demonstration cache must fail")


def test_zero_output_branch_receives_gradient_then_unlocks_inner_gradients():
    # Given a newly initialized attention branch.
    attention = DemonstrationAttention(model_dim=8, attention_dim=8, num_heads=2)
    optimizer = torch.optim.SGD(attention.parameters(), lr=0.1)
    query = torch.randn(1, 3, 8, requires_grad=True)
    demo = torch.randn(1, 2, 8, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)

    # When the first optimization step updates the zero output projection.
    attention(query, demo, mask).sum().backward()
    assert attention.output.weight.grad is not None
    assert torch.count_nonzero(attention.output.weight.grad) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    demo.grad = None

    # Then the next step propagates into demonstration K/V inputs.
    attention(query, demo, mask).sum().backward()
    assert demo.grad is not None
    assert torch.count_nonzero(demo.grad) > 0


def test_model_activation_is_serialized_and_loads_old_defaults(tmp_path):
    # Given an old-style configuration without demonstration fields.
    baseline = _model()
    old_config = dict(baseline.config)
    for key in (
        "enable_demo_conditioning",
        "demo_attention_dim",
        "demo_num_heads",
    ):
        old_config.pop(key, None)

    # When the old configuration is loaded and a new model is activated and saved.
    old_model = WanTransformer3DModel.from_config(old_config)
    assert old_model.config.enable_demo_conditioning is False
    assert not old_model.demo_conditioning_enabled
    old_model.enable_demonstration_conditioning(attention_dim=8, num_heads=2)
    old_model.save_pretrained(tmp_path)
    reloaded = WanTransformer3DModel.from_pretrained(tmp_path)

    # Then architecture settings and zero-output branches survive reload.
    assert reloaded.config.enable_demo_conditioning is True
    assert reloaded.config.demo_attention_dim == 8
    assert reloaded.config.demo_num_heads == 2
    for block in reloaded.blocks:
        assert torch.equal(
            block.demo_attention.output.weight,
            torch.zeros_like(block.demo_attention.output.weight),
        )


def test_demo_branches_exist_for_mcp_in_either_enable_order():
    # Given demonstration conditioning enabled before MCP.
    demo_first = _model()
    demo_first.enable_demonstration_conditioning(attention_dim=8, num_heads=2)

    # When MCP is subsequently enabled.
    demo_first.enable_mcp_training(1, 1, (0,))

    # Then newly created MCP blocks retain demonstration branches.
    assert hasattr(demo_first.mcp_blocks[0][0], "demo_attention")

    # Given MCP enabled before demonstration conditioning.
    mcp_first = _model(
        enable_mcp=True,
        num_mcp_depths=1,
        mcp_blocks_per_depth=1,
    )

    # When demonstration conditioning is subsequently enabled.
    mcp_first.enable_demonstration_conditioning(attention_dim=8, num_heads=2)

    # Then both main and MCP blocks receive demonstration branches.
    assert all(hasattr(block, "demo_attention") for block in mcp_first.blocks)
    assert hasattr(mcp_first.mcp_blocks[0][0], "demo_attention")


def test_model_cache_preparation_is_forward_only_and_full_clear_removes_it():
    # Given an enabled tiny model and one demonstration.
    model = _model(enable_demo_conditioning=True, demo_attention_dim=8, demo_num_heads=2)
    payload = {
        "demo_latents": torch.randn(1, 2, 2, 2, 2),
        "demo_positions": torch.tensor([[0.0, 1.0]]),
        "demo_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    # When cache preparation runs through the model forward boundary.
    sentinel = model(payload, prepare_demo_cache=True, cache_name="episode")

    # Then every block owns a persistent cache and prediction refresh preserves it.
    assert sentinel is None
    assert all(block.demo_attention.has_cache("episode") for block in model.blocks)
    model.clear_pred_cache("episode")
    assert all(block.demo_attention.has_cache("episode") for block in model.blocks)

    # When the episode cache is cleared.
    model.clear_cache("episode")

    # Then demonstration caches are removed from every block.
    assert not any(block.demo_attention.has_cache("episode") for block in model.blocks)


def test_tiny_train_forward_preserves_initial_output_and_isolates_samples():
    # Given a one-layer tiny model with a two-sample packed training input.
    model = _model(
        num_layers=1,
        enable_demo_conditioning=True,
        demo_attention_dim=8,
        demo_num_heads=2,
    ).to(dtype=torch.bfloat16)
    model.eval()
    payload = _train_input(torch.randn(2, 2, 2, 2, 2))
    without_demo = deepcopy(payload)
    without_demo.pop("demo_latents")
    without_demo.pop("demo_positions")
    without_demo.pop("demo_mask")

    # When the zero-output initialized model runs with and without a demonstration.
    no_demo_output = model(without_demo, train_mode=True)
    initial_demo_output = model(deepcopy(payload), train_mode=True)

    # Then both video and action outputs preserve pretrained behavior exactly.
    assert torch.equal(no_demo_output[0], initial_demo_output[0])
    assert torch.equal(no_demo_output[1], initial_demo_output[1])

    # Given a nonzero trained demonstration output projection.
    with torch.no_grad():
        for block in model.blocks:
            block.demo_attention.output.weight.fill_(0.1)
    changed_payload = deepcopy(payload)
    changed_payload["demo_latents"][1] += 100

    # When only sample one's demonstration changes.
    baseline = model(deepcopy(payload), train_mode=True)
    changed = model(changed_payload, train_mode=True)

    # Then packed sample zero's video and action predictions stay isolated.
    assert torch.allclose(baseline[0][0], changed[0][0])
    assert torch.allclose(baseline[1][0], changed[1][0])
    assert not torch.allclose(baseline[0][1], changed[0][1])


def test_tiny_train_forward_propagates_main_and_mcp_demo_gradients():
    # Given a tiny MCP model with demonstration conditioning in every block.
    model = _model(
        num_layers=1,
        enable_mcp=True,
        num_mcp_depths=1,
        mcp_blocks_per_depth=1,
        enable_demo_conditioning=True,
        demo_attention_dim=8,
        demo_num_heads=2,
    ).to(dtype=torch.bfloat16)
    payload = _train_input(torch.randn(2, 2, 2, 2, 2), include_mcp=True)

    # When video, action, and MCP outputs contribute to one training loss.
    video_output, action_output, mcp_outputs = model(payload, train_mode=True)
    loss = video_output.float().sum() + action_output.float().sum()
    loss = loss + sum(output.float().sum() for output in mcp_outputs)
    loss.backward()

    # Then every zero-output demonstration branch participates in backpropagation.
    branch_gradients = [
        block.demo_attention.output.weight.grad for block in model.blocks
    ] + [
        block.demo_attention.output.weight.grad
        for group in model.mcp_blocks
        for block in group
    ]
    assert all(gradient is not None for gradient in branch_gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in branch_gradients)


def test_training_never_reuses_a_stale_inference_demo_cache():
    # Given a trained demo branch, a prepared cache, and a no-demo training batch.
    model = _model(
        num_layers=1,
        enable_demo_conditioning=True,
        demo_attention_dim=8,
        demo_num_heads=2,
    ).to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.blocks[0].demo_attention.output.weight.fill_(0.1)
    payload = _train_input(torch.randn(2, 2, 2, 2, 2))
    cache_payload = {
        key: payload[key]
        for key in ("demo_latents", "demo_positions", "demo_mask")
    }
    without_demo = deepcopy(payload)
    for key in cache_payload:
        without_demo.pop(key)
    uncached = model(deepcopy(without_demo), train_mode=True)
    model(cache_payload, prepare_demo_cache=True, cache_name="episode")

    # When training runs after inference cache preparation without demo tensors.
    after_cache = model(deepcopy(without_demo), train_mode=True)

    # Then the stale cached demonstration cannot affect either training output.
    assert torch.equal(uncached[0], after_cache[0])
    assert torch.equal(uncached[1], after_cache[1])
