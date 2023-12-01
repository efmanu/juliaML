using Flux
using Random
using Flux.Data: DataLoader

# Generate synthetic data for linear regression
X = rand(Float32, 1,100)
y = X[1,:].*3 .+ 2 + randn(100, 1).*0.1
Y = reshape(y,1, length(y))

# Splitting the dataset into training and testing sets
data = DataLoader((X,Y); batchsize=2, shuffle=true)

# Define a simple linear regression model
model = Chain(Dense(1, 10),Dense(10, 1))

# Define the loss function and optimizer
loss(x, y) = Flux.mse(model(x), y)
optimizer = Flux.Descent()

# Training the model
for epoch in 1:200
    Flux.train!(loss, Flux.params(model), data, optimizer)
    @info "Epoch $epoch and loss $(loss(X,Y))"
end

# Make predictions
X_new = [1.0, 2.0, 3.0]
prediction =model(reshape(X_new,1, length(X_new)))
println("Predictions for $X_new: $prediction")