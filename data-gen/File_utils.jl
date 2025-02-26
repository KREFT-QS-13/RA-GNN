module Filul

using Glob

export check_exists, CustomError

# Function to find the largest number in filenames
function find_largest_integer(filenames)::Int
    max_number = 0 #-1
    regex = r"[0-9]+"
    for filename in filenames
        for match in eachmatch(regex, filename)
            current_number = parse(Int, match.match)
            if current_number > max_number
                max_number = current_number
            end
        end
    end
    return max_number
end

function check_exists(folder_name::String)::Int
    largest_int = 0
    if !isdir(folder_name)
        mkpath(folder_name)
    else # get the largest realization number
        files_in_dir = glob("Mg_*.npy",folder_name)
        files_in_dir = [split(ff,"/")[end] for ff in files_in_dir]
        largest_int = find_largest_integer(files_in_dir)
        println("largest_int = ", largest_int)
    end
    return largest_int
end

struct CustomError <: Exception
    msg::String
end

function Base.showerror(io::IO, e::CustomError)
    print(io, e.msg)
end

end