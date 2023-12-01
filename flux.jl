using Flux
using Flux.Data: DataLoader
using Random
using Statistics

# Example: Creating a simple dataset
X = rand(Float32, 2, 100)  # Features
y = X[1,:].*2+X[2,:].^2   # Labels
Y = reshape(y,1, length(y))
# Splitting the dataset into training and testing sets
data = DataLoader((X, Y), batchsize=2, shuffle=true)

model = Chain(
    Dense(2, 10, σ),   # Input layer: 2 features, 10 nodes, σ activation function
    Dense(10, 1)       # Output layer: 1 node (for regression tasks)
)

# Defining a loss function and an optimizer
loss(model, x, y) = mean(abs2.(model(x) .- y));
opt = Descent()


for epoch in 1:200
    @info "Epoch $epoch and loss $(loss(model, X,Y))"
    Flux.train!(loss, model, data, opt)
end