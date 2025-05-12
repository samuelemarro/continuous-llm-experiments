from transformers import PhiPreTrainedModel
import torch

# Generates a causal mask that takes into account the positions of the tokens
# Not suitable for sequences where the positions are non-unique or non-monotonic
def get_continuity_mask(positions, integration_start=0, device='cuda'):
    for i in range(len(positions) - 1):
        if positions[i] >= positions[i + 1]:
            raise ValueError("The positions must be in increasing order")

    seqlen = len(positions)
    multiplicative_mask = torch.zeros((1, 1, seqlen, seqlen), device=device)
    # multiplicative_mask[i, j] = 1 if j is used to predict i, 0 otherwise
    # Note: with the current implementation, it's also possible to have other values

    for i in range(seqlen):
        for j in range(0, seqlen):
            source_position = positions[j] # The position we are looking at
            target_position = positions[i] # The position we are trying to predict

            if target_position < source_position:
                # We can't look into the future
                multiplicative_mask[0, 0, i, j] = 0
                continue

            # Rectangle integration: the weight of source_position is equal
            # to how much "time" (delta_x) has passed since the last time
            # we saw another position (previous_position)

            all_previous_positions = positions[positions < source_position]

            if len(all_previous_positions) == 0:
                previous_position = integration_start
            else:
                previous_position = torch.max(all_previous_positions)
            #print('Target:', target_position, 'Previous:', previous_position)

            delta_x = source_position - previous_position

            multiplicative_mask[0, 0, i, j] = delta_x

    # The transformer accepts an additive mask for the logit, so we compute the additive mask
    # as the log of the multiplicative mask, due to the fact that:
    # multiplicative_mask * exp(logit) = exp(logit + log(multiplicative_mask))
    mask = torch.log(multiplicative_mask + 1e-50)

    return mask

def get_outputs_from_continuous_inputs(model, embeddings=None, input_ids=None, position_ids=None, integration_start=0, return_dict=False):
    assert (embeddings is not None) != (input_ids is not None) # xor
    if embeddings is None:
        embeddings = model.get_input_embeddings()(input_ids)
        #print(embeddings.sum(dim=-1))
    
    if position_ids is None:
        position_ids = torch.arange(embeddings.shape[1], device=embeddings.device) + integration_start + 1


    attention_mask = get_continuity_mask(position_ids, integration_start=integration_start, device='cpu')
    #print('Attention mask:', attention_mask)
    # Temp
    #position_ids -= 1

    extra_kwargs = {}

    if isinstance(model, PhiPreTrainedModel):
        extra_kwargs['trust_remote_code'] = True
    
    print(extra_kwargs)

    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask, position_ids=torch.unsqueeze(position_ids, dim=0), return_dict=return_dict,
                    use_cache=False, **extra_kwargs)
    return outputs