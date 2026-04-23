import numpy as np
from model import get_model
from blockchain import Blockchain

# Initialize blockchain
blockchain = Blockchain()

def aggregate(client_weights, client_sizes):
    total = sum(client_sizes)

    # Federated Averaging
    avg_coef = sum(w * (size / total) for (w, _), size in zip(client_weights, client_sizes))
    avg_intercept = sum(b * (size / total) for (_, b), size in zip(client_weights, client_sizes))

    # 🔐 Store model update in blockchain
    block_data = {
        "weights": avg_coef.tolist(),
        "intercept": avg_intercept.tolist(),
        "num_clients": len(client_weights)
    }

    blockchain.add_block(block_data)

    print("\n🔗 Block added to Blockchain!")
    print(f"Block Index: {len(blockchain.chain)-1}")
    print(f"Data: {block_data}\n")

    # Create global model
    model = get_model()
    model.coef_ = avg_coef
    model.intercept_ = avg_intercept

    return model