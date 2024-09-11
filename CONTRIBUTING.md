# Contributing to NuMojo

Thank you for your interest in contributing to NuMojo! We appreciate your efforts to make this project better. Below are some guidelines to help you contribute effectively.

## Style Guide

Please follow the Mojo standard library style guide for all contributions. Consistency in code style helps maintain readability and ease of collaboration. Key points include:

- Use clear, descriptive names for variables and functions.
- Write concise, well-documented code.
- Adhere to formatting conventions for indentation, spacing, and line breaks.

Additionally refer to `style guide.md` for docstring and naming conventions.

## Pull Requests

When submitting pull requests:

- Ensure they are small but complete. A pull request should introduce at least a minimally viable version of the feature you are adding.
- Include tests for your contributions where applicable.
- Provide a clear and concise description of what your pull request accomplishes.

## Just Do It

If you have an idea or want to work on something, go ahead and do it! You don’t need to ask for permission before starting. In fact, we prefer if you avoid “licking the cookie” by claiming tasks without following through. We would rather recieve 5 different ways of accomplishing something and then choose the best one or combine then than not recieve any feature at all.

## Directory Structure

Organize your additions into the appropriate submodule or file, if one does not exist feel free to make it and we can figure out where it goes during the pull request checks. This helps keep the project structured and maintainable. For example:

- If you’re adding a statistics function, place it in the `stats` submodule.
  - If a stats module does not yet exist put the code in a directory called stats in a file with a name that describes the sub disipline of statistics the code enables, along with a `__init__.mojo`
  - For a kernel density estimation function, add it to the `kde.mojo` file within the `stats` directory.

Following this structure ensures that similar functionalities are grouped together, making the codebase easier to navigate.

## Contribution Process

1. **Fork the Repository**: Create a personal fork of the repository on GitHub.
2. **Clone Your Fork**: Clone your forked repository to your local machine.

   ```sh
   git clone https://github.com/your-username/numojo.git
   ```

3. **Create a Branch**: Create a new branch for your feature or bugfix.

   ```sh
   git checkout -b feature-name
   ```

4. **Make Your Changes**: Implement your changes in your branch.
5. **Run Tests**: NuMojo now uses the `Magic` package manager by Modular. To ensure that all unit tests pass, the NuMojo module packages correctly, and the .mojo files are properly formatted, run the following command:
   ```sh
   magic run final
   ```
6. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.

   ```sh
   git commit -m "Add feature XYZ"
   ```

7. **Push Your Changes**: Push your branch to your fork on GitHub.

   ```sh
   git push origin feature-name
   ```

8. **Submit a Pull Request**: Open a pull request to the `main` branch of the original repository.
