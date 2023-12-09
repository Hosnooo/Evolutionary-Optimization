%% Plotting Enhancement
function newLimits = edit_limits(axis, limitIncreaseFactor)
    currentLimits = axis;
    evenIndices = 2:2:numel(currentLimits);
    oddIndices = 1:2:numel(currentLimits);
    increase_arr = repelem(currentLimits(evenIndices), 2);
    increase_arr(oddIndices) = -increase_arr(oddIndices);
    newLimits = currentLimits + limitIncreaseFactor * increase_arr;
end